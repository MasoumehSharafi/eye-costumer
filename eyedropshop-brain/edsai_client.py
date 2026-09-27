import json
import os
import re
import time
import unicodedata
from typing import Any, Dict, List, Optional
import random
import threading

import requests
from dotenv import load_dotenv


load_dotenv()


BASE_URL = os.getenv(
    "EDSAI_BASE_URL",
    "https://edsai-api.daniel-ed7.workers.dev",
).rstrip("/")

RENEWAL_CREDENTIAL = os.getenv(
    "EDSAI_CENTRAL_BRAIN_RENEWAL_KEY"
)


_session = requests.Session()

_cached_token: Optional[str] = None
_token_expires_at: float = 0.0
_token_lock = threading.Lock()

# ---------------------------------------------------------
# Authentication
# ---------------------------------------------------------

def _sleep_before_retry(
    attempt: int,
    rate_limited: bool = False,
) -> None:
    """
    Bounded exponential backoff with jitter.

    edsAI asks clients to wait roughly one minute after 429.
    """
    if rate_limited:
        delay = 60 + random.uniform(0, 5)
    else:
        delay = (2 ** attempt) + random.uniform(0, 1)

    time.sleep(delay)


def _exchange_renewal_credential():
    """
    Exchange the long-lived renewal credential for a short-lived
    edsAI access token.

    Handles temporary network/server failures with bounded retries.
    """
    max_attempts = 3

    for attempt in range(max_attempts):
        try:
            response = _session.post(
                f"{BASE_URL}/v1/auth/token",
                headers={
                    "Authorization": (
                        f"Bearer {RENEWAL_CREDENTIAL}"
                    ),
                },
                timeout=30,
            )

        except (
            requests.Timeout,
            requests.ConnectionError,
        ) as exc:
            if attempt < max_attempts - 1:
                _sleep_before_retry(attempt)
                continue

            raise RuntimeError(
                "edsAI token renewal failed after "
                "repeated network errors."
            ) from exc

        if response.status_code == 401:
            raise RuntimeError(
                "edsAI renewal credential was rejected. "
                "Check the credential/environment "
                "or contact Daniel."
            )

        if response.status_code == 403:
            raise RuntimeError(
                "edsAI Central Brain profile is "
                "not permitted to renew access."
            )

        if response.status_code == 409:
            raise RuntimeError(
                "edsAI access state changed or "
                "the token limit was reached."
            )

        if response.status_code == 429:
            if attempt < max_attempts - 1:
                _sleep_before_retry(
                    attempt,
                    rate_limited=True,
                )
                continue

            raise RuntimeError(
                "edsAI token renewal rate limit reached."
            )

        if response.status_code in {
            500,
            502,
            503,
            504,
        }:
            if attempt < max_attempts - 1:
                _sleep_before_retry(attempt)
                continue

            raise RuntimeError(
                "edsAI token service remained "
                "unavailable after retries."
            )

        response.raise_for_status()

        data = response.json()

        access_token = data.get("access_token")
        expires_at = data.get("expires_at")

        if not access_token or not expires_at:
            raise RuntimeError(
                "edsAI token response did not contain "
                "access_token/expires_at."
            )

        return (
            access_token,
            float(expires_at),
        )

    raise RuntimeError(
        "edsAI token renewal failed."
    )


def get_access_token(
    force_refresh: bool = False,
    invalid_token: Optional[str] = None,
) -> str:
    """
    Return a cached access token when possible.

    Renew when fewer than five minutes remain.

    Concurrent renewal requests are coalesced so several Brain
    requests do not unnecessarily generate several access tokens.
    """
    global _cached_token
    global _token_expires_at

    if not RENEWAL_CREDENTIAL:
        raise RuntimeError(
            "EDSAI_CENTRAL_BRAIN_RENEWAL_KEY is not set."
        )

    now = time.time()

    if (
        not force_refresh
        and _cached_token
        and _token_expires_at > now + 300
    ):
        return _cached_token

    with _token_lock:
        now = time.time()

        # Another request may have renewed the token while
        # this request was waiting for the lock.
        if (
            invalid_token is not None
            and _cached_token
            and _cached_token != invalid_token
            and _token_expires_at > now + 300
        ):
            return _cached_token

        # Double-check after obtaining the lock.
        if (
            not force_refresh
            and _cached_token
            and _token_expires_at > now + 300
        ):
            return _cached_token

        # The server rejected this particular cached token.
        if (
            invalid_token is not None
            and _cached_token == invalid_token
        ):
            _cached_token = None
            _token_expires_at = 0.0

        access_token, expires_at = (
            _exchange_renewal_credential()
        )

        _cached_token = access_token
        _token_expires_at = expires_at

        return _cached_token


# ---------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------

def _request_data(
    path: str,
    params: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Perform an authenticated edsAI GET.

    Behaviour:
    - renew and retry once after 401
    - stop immediately on 403
    - surface 409 without endless retries
    - back off with jitter after 429
    - bounded retries for 5xx
    - bounded retries for timeout/connection failures
    """

    max_attempts = 3
    attempt = 0
    retried_after_401 = False

    while attempt < max_attempts:
        token = get_access_token()

        try:
            response = _session.get(
                f"{BASE_URL}{path}",
                headers={
                    "Authorization": f"Bearer {token}",
                },
                params=params or {},
                timeout=30,
            )

        except (
            requests.Timeout,
            requests.ConnectionError,
        ) as exc:
            if attempt < max_attempts - 1:
                _sleep_before_retry(attempt)
                attempt += 1
                continue

            raise RuntimeError(
                f"edsAI request failed after "
                f"network retries: {path}"
            ) from exc

        # The access token itself failed.
        # Renew exactly once and retry the request.
        if response.status_code == 401:
            if retried_after_401:
                raise RuntimeError(
                    "edsAI request still returned "
                    "401 after token renewal."
                )

            retried_after_401 = True

            get_access_token(
                force_refresh=True,
                invalid_token=token,
            )

            continue

        # Permission/configuration problem.
        # Do not retry indefinitely.
        if response.status_code == 403:
            raise RuntimeError(
                f"edsAI permission denied for {path}."
            )

        # Access state changed or token limit condition.
        if response.status_code == 409:
            raise RuntimeError(
                "edsAI access state changed or "
                "the token limit was reached."
            )

        # Rate limiting.
        if response.status_code == 429:
            if attempt < max_attempts - 1:
                _sleep_before_retry(
                    attempt,
                    rate_limited=True,
                )
                attempt += 1
                continue

            raise RuntimeError(
                "edsAI rate limit reached after retries."
            )

        # Temporary upstream/service problem.
        if response.status_code in {
            500,
            502,
            503,
            504,
        }:
            if attempt < max_attempts - 1:
                _sleep_before_retry(attempt)
                attempt += 1
                continue

            raise RuntimeError(
                f"edsAI service remained unavailable "
                f"after retries: {path}"
            )

        response.raise_for_status()

        return response.json()

    raise RuntimeError(
        f"edsAI request failed after retries: {path}"
    )

# ---------------------------------------------------------
# Pagination
# ---------------------------------------------------------

def _get_all_pages(path: str) -> List[Dict[str, Any]]:
    """
    Read every page until hasMore is False.

    Important: do NOT stop because a page contains zero items.
    edsAI may return an empty filtered page while hasMore=True.
    """

    all_items: List[Dict[str, Any]] = []
    cursor: Optional[str] = None

    seen_cursors = set()
    while True:
        if len(seen_cursors) >= 1000:
            raise RuntimeError("edsAI pagination exceeded page limit")
        params = {}

        if cursor:
            params["cursor"] = cursor

        data = _request_data(
            path,
            params=params,
        )

        items = data.get("items", [])

        if isinstance(items, list):
            all_items.extend(items)

        has_more = bool(data.get("hasMore", False))

        if not has_more:
            break

        cursor = data.get("nextCursor")

        if cursor in seen_cursors:
            raise RuntimeError("edsAI returned a repeated pagination cursor")
        seen_cursors.add(cursor)
        if not cursor:
            raise RuntimeError(
                f"{path} returned hasMore=True "
                "but did not provide nextCursor."
            )

    return all_items


def get_all_products() -> List[Dict[str, Any]]:
    return _get_all_pages("/v1/products")


def get_all_inventory() -> List[Dict[str, Any]]:
    return _get_all_pages("/v1/inventory")


# ---------------------------------------------------------
# Shopify rich-text parsing
# ---------------------------------------------------------

def _extract_rich_text(node: Any) -> List[str]:
    """
    Recursively extract readable text from Shopify rich-text JSON.
    """

    output: List[str] = []

    if isinstance(node, dict):
        node_type = node.get("type")

        if node_type == "text":
            value = node.get("value")

            if value:
                output.append(str(value))

        children = node.get("children")

        if isinstance(children, list):
            child_text: List[str] = []

            for child in children:
                child_text.extend(_extract_rich_text(child))

            if node_type in {
                "paragraph",
                "list-item",
                "heading",
            }:
                if child_text:
                    output.append("".join(child_text).strip())
            else:
                output.extend(child_text)

    elif isinstance(node, list):
        for child in node:
            output.extend(_extract_rich_text(child))

    return output


def parse_metafield_value(
    field_type: Optional[str],
    value: Any,
) -> Any:
    """
    Convert Shopify metafield values into model-friendly data.
    """

    if value is None:
        return None

    if field_type != "rich_text_field":
        return value

    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value

    parts = _extract_rich_text(parsed)

    # Remove empty and duplicate adjacent parts
    cleaned: List[str] = []

    for part in parts:
        part = re.sub(r"\s+", " ", part).strip()

        if not part:
            continue

        if not cleaned or cleaned[-1] != part:
            cleaned.append(part)

    return "\n".join(cleaned)


def clean_metafields(
    metafields: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Flatten metafields into convenient names.

    Example:
        custom.ingredients -> ingredients
        custom.more_details -> more_details
    """

    result: Dict[str, Any] = {}

    for field in metafields or []:
        namespace = field.get("namespace")
        key = field.get("key")
        field_type = field.get("type")
        value = field.get("value")

        if not key:
            continue

        parsed_value = parse_metafield_value(
            field_type,
            value,
        )

        full_key = (
            f"{namespace}.{key}"
            if namespace
            else key
        )

        # Preserve fully-qualified key
        result[full_key] = parsed_value

        # Convenient short key
        if key not in result:
            result[key] = parsed_value

    return result


# ---------------------------------------------------------
# Product + inventory join
# ---------------------------------------------------------

def join_products_inventory(
    products: List[Dict[str, Any]],
    inventory: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Join inventory to product variants using Shopify variant ID.
    """

    inventory_by_variant = {
        row.get("variantId"): row
        for row in inventory
        if row.get("variantId")
    }

    joined: List[Dict[str, Any]] = []

    for product in products:
        variant_id = product.get("id")
        inventory_row = inventory_by_variant.get(
            variant_id,
            {},
        )

        cleaned = {
            "variant_id": variant_id,
            "product_id": product.get("productId"),
            "title": product.get("title"),
            "variant_title": product.get("variantTitle"),
            "description": product.get("description"),
            "vendor": product.get("vendor"),
            "product_type": product.get("productType"),
            "sku": product.get("sku"),
            "barcode": product.get("barcode"),

            # Currency is deliberately NOT added here.
            "retail_price": product.get("retailPrice"),

            "image": product.get("image"),

            "tracked": inventory_row.get("tracked"),
            "available": inventory_row.get("available"),

            "metafields": clean_metafields(
                product.get("metafields", [])
            ),
        }

        joined.append(cleaned)

    return joined


def get_catalog(
    include_inventory: bool = True,
) -> List[Dict[str, Any]]:
    products = get_all_products()

    if not include_inventory:
        return [
            {
                "variant_id": product.get("id"),
                "product_id": product.get("productId"),
                "title": product.get("title"),
                "variant_title": product.get("variantTitle"),
                "description": product.get("description"),
                "vendor": product.get("vendor"),
                "product_type": product.get("productType"),
                "sku": product.get("sku"),
                "barcode": product.get("barcode"),
                "retail_price": product.get("retailPrice"),
                "image": product.get("image"),
                "metafields": clean_metafields(
                    product.get("metafields", [])
                ),
            }
            for product in products
        ]

    inventory = get_all_inventory()

    return join_products_inventory(
        products,
        inventory,
    )


# ---------------------------------------------------------
# Product matching
# ---------------------------------------------------------

def _normalize(text: Optional[str]) -> str:
    if not text:
        return ""

    text = unicodedata.normalize(
        "NFKD",
        str(text),
    )

    text = "".join(
        char
        for char in text
        if not unicodedata.combining(char)
    )

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def find_products(
    query: str,
    catalog: Optional[List[Dict[str, Any]]] = None,
    include_inventory: bool = True,
) -> List[Dict[str, Any]]:
    """
    Simple deterministic product matcher.

    Search:
    title
    variant title
    SKU
    barcode
    vendor
    """

    if catalog is None:
        catalog = get_catalog(
            include_inventory=include_inventory
        )

    normalized_query = _normalize(query)

    if not normalized_query:
        return []

    query_tokens = normalized_query.split()

    matches = []

    for item in catalog:
        searchable = " ".join(
            [
                str(item.get("title") or ""),
                str(item.get("variant_title") or ""),
                str(item.get("sku") or ""),
                str(item.get("barcode") or ""),
                str(item.get("vendor") or ""),
            ]
        )

        normalized_searchable = _normalize(searchable)

        # Exact phrase gets priority
        exact_phrase = (
            normalized_query
            in normalized_searchable
        )

        # Otherwise require all words
        all_tokens = all(
            token in normalized_searchable
            for token in query_tokens
        )

        if exact_phrase or all_tokens:
            matches.append(
                (
                    0 if exact_phrase else 1,
                    item,
                )
            )

    matches.sort(
        key=lambda x: x[0]
    )

    return [
        item
        for _, item in matches
    ]


def find_product(
    query: str,
    include_inventory: bool = True,
) -> Optional[Dict[str, Any]]:
    matches = find_products(
        query,
        include_inventory=include_inventory,
    )

    return matches[0] if matches else None