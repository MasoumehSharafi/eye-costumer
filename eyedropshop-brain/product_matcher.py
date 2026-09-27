from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
import unicodedata

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PRODUCT_FILE = BASE_DIR / "prototype_data/products/products_30_bilingual.csv"


@dataclass(frozen=True)
class ProductMatch:
    product_handle: str
    product_title: str
    matched_alias: str


@dataclass(frozen=True)
class ProductAmbiguity:
    matched_alias: str
    product_titles: tuple[str, ...]


def normalize_text(text: str) -> str:
    """Normalize user text and product names for phrase matching."""
    text = unicodedata.normalize("NFKD", str(text).casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _candidate_aliases(title: str, handle: str) -> set[str]:
    aliases: set[str] = set()

    title_normalized = normalize_text(title)
    if title_normalized:
        aliases.add(title_normalized)

        words = title_normalized.split()

        # Useful multi-word prefixes such as "refresh optive".
        for end in range(2, len(words)):
            prefix = " ".join(words[:end])
            if len(prefix) >= 8:
                aliases.add(prefix)

        # Useful one-word brand/family names such as Refresh, CALMO,
        # Soothe, or Cheshma. Generic eye/eyes tokens are intentionally
        # excluded.
        if words:
            first_word = words[0]
            if (
                len(first_word) >= 5
                and first_word not in {"eye", "eyes"}
            ):
                aliases.add(first_word)

    # Titles using "|" often have a meaningful short name after it.
    for part in re.split(r"[|]", str(title)):
        normalized_part = normalize_text(part)
        if len(normalized_part) >= 6:
            aliases.add(normalized_part)

    handle_alias = normalize_text(str(handle).replace("-", " "))
    if len(handle_alias) >= 5:
        aliases.add(handle_alias)

    return aliases


@lru_cache(maxsize=1)
def load_product_index():
    products = pd.read_csv(PRODUCT_FILE)

    product_info: dict[str, dict] = {}
    alias_to_handles: dict[str, set[str]] = {}

    for handle, group in products.groupby("product_handle", sort=False):
        handle = str(handle)

        titles_by_language: dict[str, str] = {}
        all_titles: list[str] = []

        for _, row in group.iterrows():
            language = str(row.get("language", "")).strip()
            title_value = row.get("title")

            if pd.notna(title_value) and str(title_value).strip():
                title = str(title_value).strip()
                titles_by_language[language] = title
                all_titles.append(title)

        fallback_title = (
            titles_by_language.get("en")
            or next(iter(titles_by_language.values()), handle)
        )

        product_info[handle] = {
            "titles": titles_by_language,
            "fallback_title": fallback_title,
        }

        aliases: set[str] = set()

        for title in all_titles:
            aliases.update(_candidate_aliases(title, handle))

        aliases.update(_candidate_aliases("", handle))

        for alias in aliases:
            alias_to_handles.setdefault(alias, set()).add(handle)

    unique_aliases = {
        alias: next(iter(handles))
        for alias, handles in alias_to_handles.items()
        if len(handles) == 1
    }

    ambiguous_aliases = {
        alias: tuple(sorted(handles))
        for alias, handles in alias_to_handles.items()
        if len(handles) > 1
    }

    ordered_unique_aliases = sorted(
        unique_aliases.items(),
        key=lambda item: (
            len(item[0].split()),
            len(item[0]),
        ),
        reverse=True,
    )

    ordered_ambiguous_aliases = sorted(
        ambiguous_aliases.items(),
        key=lambda item: (
            len(item[0].split()),
            len(item[0]),
        ),
        reverse=True,
    )

    return (
        product_info,
        ordered_unique_aliases,
        ordered_ambiguous_aliases,
    )


def detect_product_mentions(
    question: str,
    language: str,
) -> list[ProductMatch]:
    """
    Return every uniquely identifiable product explicitly mentioned.

    A comparison such as "Refresh Tears or Refresh Optive Advanced"
    therefore returns two products instead of silently selecting one.
    """
    normalized_question = normalize_text(question)

    if not normalized_question:
        return []

    padded_question = f" {normalized_question} "

    product_info, ordered_unique_aliases, _ = load_product_index()

    matches: list[ProductMatch] = []
    seen_handles: set[str] = set()

    for alias, handle in ordered_unique_aliases:
        if handle in seen_handles:
            continue

        if f" {alias} " not in padded_question:
            continue

        info = product_info[handle]
        title = (
            info["titles"].get(language)
            or info["titles"].get("en")
            or info["fallback_title"]
        )

        matches.append(
            ProductMatch(
                product_handle=handle,
                product_title=title,
                matched_alias=alias,
            )
        )
        seen_handles.add(handle)

    return matches


def detect_exact_product(
    question: str,
    language: str,
) -> ProductMatch | None:
    """
    Return a product only when exactly one product is uniquely mentioned.

    If two products are named for a comparison, return None rather than
    pretending the question is about only one of them.
    """
    matches = detect_product_mentions(question, language)

    if len(matches) == 1:
        return matches[0]

    return None


def detect_ambiguous_product(
    question: str,
    language: str,
) -> ProductAmbiguity | None:
    """
    Detect an ambiguous family/brand reference such as "Refresh".

    Exact product names take precedence. If one or more specific products
    are already named, this function returns None.
    """
    if detect_product_mentions(question, language):
        return None

    normalized_question = normalize_text(question)

    if not normalized_question:
        return None

    padded_question = f" {normalized_question} "

    product_info, _, ordered_ambiguous_aliases = load_product_index()

    for alias, handles in ordered_ambiguous_aliases:
        if f" {alias} " not in padded_question:
            continue

        titles = []

        for handle in handles:
            info = product_info[handle]
            title = (
                info["titles"].get(language)
                or info["titles"].get("en")
                or info["fallback_title"]
            )
            titles.append(title)

        return ProductAmbiguity(
            matched_alias=alias,
            product_titles=tuple(sorted(set(titles))),
        )

    return None