from __future__ import annotations

import re
import time
import unicodedata
from typing import Any, Dict, List, Optional

from edsai_client import (
    get_all_products,
    get_all_inventory,
    clean_metafields,
)

from edsai_intelligence import (
    understand_query,
    semantic_candidates,
    rerank_candidates,
)
# ---------------------------------------------------------
# Small in-process caches
# ---------------------------------------------------------

# Products do not normally need to be downloaded again for
# every customer question.
PRODUCT_CACHE_TTL = 300  # 5 minutes

# Inventory is more time-sensitive.
INVENTORY_CACHE_TTL = 30  # 30 seconds


_product_cache: List[Dict[str, Any]] = []
_product_cache_time: float = 0.0

_inventory_cache: List[Dict[str, Any]] = []
_inventory_cache_time: float = 0.0

def _current_question_text(question: str) -> str:
    marker = "Current question:"

    if marker in question:
        return question.rsplit(
            marker,
            1,
        )[1].strip()

    return question.strip()

def _is_personal_symptom_recommendation(
    question: str,
    language: str,
) -> bool:
    q = _normalize(question)

    personal_symptoms = (
        "my eye",
        "my eyes",
        "i have",
        "i am experiencing",
        "i'm experiencing",
    )

    recommendation_intent = (
        "which eye drops",
        "what eye drops",
        "which product",
        "what product",
        "should i buy",
        "should i use",
        "do you recommend",
        "recommend for me",
    )

    return (
        any(x in q for x in personal_symptoms)
        and any(x in q for x in recommendation_intent)
    )

# ---------------------------------------------------------
# Normalization
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

    return " ".join(text.split())

HIGH_RISK_EYE_TERMS = (
    # Pain
    "eye pain",
    "pain in my eye",
    "pain in my eyes",
    "painful eye",
    "painful eyes",
    "eye is painful",
    "eyes are painful",
    "eye hurts",
    "eyes hurt",
    "severe pain",
    "painful",

    # Vision changes
    "blurry vision",
    "blurred vision",
    "vision is blurry",
    "vision is blurred",
    "blurry",
    "blurred",
    "vision loss",
    "loss of vision",
    "sudden vision change",
    "sudden change in vision",
    "changes in vision",

    # Injury / chemical exposure
    "eye injury",
    "injured my eye",
    "injured my eyes",
    "hit my eye",
    "chemical in my eye",
    "chemical in my eyes",
    "chemical burn",
    "foreign body",
    "something stuck in my eye",

    # Infection / discharge
    "eye infection",
    "infected eye",
    "infected eyes",
    "pus",
    "eye discharge",
    "discharge from my eye",
    "discharge from my eyes",

    # Severe swelling
    "severe eye swelling",
    "severe swelling",
    "eye swollen shut",
    "eyes swollen shut",

    # Important pre-existing condition
    "glaucoma",
)


HIGH_RISK_RECOMMENDATION_TERMS = (
    "which eye drops",
    "what eye drops",
    "which product",
    "what product",
    "what should i use",
    "should i buy",
    "should i use",
    "can i use",
    "do you recommend",
    "recommend for me",
    "is it safe for me",
    "is this safe for me",
)


def _is_high_risk_eye_recommendation(
    question: str,
    language: str,
) -> bool:
    q = _normalize(
        question
    )

    has_high_risk_context = any(
        term in q
        for term in HIGH_RISK_EYE_TERMS
    )

    has_recommendation_intent = any(
        term in q
        for term in HIGH_RISK_RECOMMENDATION_TERMS
    )

    return (
        has_high_risk_context
        and has_recommendation_intent
    )
# ---------------------------------------------------------
# Cached API reads
# ---------------------------------------------------------

def _products(
    force_refresh: bool = False,
) -> List[Dict[str, Any]]:
    global _product_cache
    global _product_cache_time

    now = time.time()

    if (
        not force_refresh
        and _product_cache
        and now - _product_cache_time < PRODUCT_CACHE_TTL
    ):
        return _product_cache

    _product_cache = get_all_products()
    _product_cache_time = now

    return _product_cache


def _inventory(
    force_refresh: bool = False,
) -> List[Dict[str, Any]]:
    global _inventory_cache
    global _inventory_cache_time

    now = time.time()

    if (
        not force_refresh
        and _inventory_cache
        and now - _inventory_cache_time < INVENTORY_CACHE_TTL
    ):
        return _inventory_cache

    _inventory_cache = get_all_inventory()
    _inventory_cache_time = now

    return _inventory_cache


# ---------------------------------------------------------
# Question intent
# ---------------------------------------------------------

def _asks_inventory(
    question: str,
    language: str,
) -> bool:
    q = _normalize(question)

    if language == "fr":
        terms = (
            "disponible",
            "disponibilite",
            "en stock",
            "stock",
            "inventaire",
            "avez vous",
        )
    else:
        terms = (
            "available",
            "availability",
            "in stock",
            "stock",
            "inventory",
            "do you have",
            "have any",
        )

    return any(
        term in q
        for term in terms
    )


def _asks_price(
    question: str,
    language: str,
) -> bool:
    q = _normalize(question)

    if language == "fr":
        terms = (
            "prix",
            "combien",
            "coute",
            "cout",
        )
    else:
        terms = (
            "price",
            "how much",
            "cost",
        )

    return any(
        term in q
        for term in terms
    )


def _requested_sections(
    question: str,
    language: str,
) -> set[str]:
    q = _normalize(question)

    sections = set()

    if language == "fr":
        mapping = {
            "ingredients": (
                "ingredient",
                "ingredients",
                "composition",
                "contient",
                "conservateur",
                "conservateurs",
                "agent de conservation",
                "agents de conservation",
            ),
            "warnings": (
                "avertissement",
                "avertissements",
                "precaution",
                "precautions",
                "mise en garde",
            ),
            "usage": (
                "comment utiliser",
                "mode d emploi",
                "utilisation",
                "dosage",
                "dose",
            ),
            "benefits_and_details": (
                "avantage",
                "avantages",
                "benefice",
                "benefices",
                "caracteristique",
            ),
        }
    else:
        mapping = {
            "ingredients": (
                "ingredient",
                "ingredients",
                "contains",
                "contain",
                "made of",
                "preservative",
                "preservatives",
                "preservative free",
            ),
            "warnings": (
                "warning",
                "warnings",
                "caution",
                "precaution",
            ),
            "usage": (
                "how to use",
                "how do i use",
                "how should i use",
                "how can i use",
                "how to apply",
                "how should i apply",
                "directions",
                "dosage",
                "dose",
                "instructions",
            ),
            "benefits_and_details": (
                "benefit",
                "benefits",
                "feature",
                "features",
                "details",
            ),
        }

    # Broad named-product questions should retrieve a concise product overview
    # instead of every available field (ingredients, warnings, usage, expiry, ...).
    generic_description_terms = (
        "what is",
        "what are",
        "tell me about",
        "describe",
        "c est quoi",
        "qu est ce que",
        "parle moi de",
        "decris",
    )

    if any(term in q for term in generic_description_terms):
        sections.add("description")

    for section, terms in mapping.items():
        if any(
            term in q
            for term in terms
        ):
            sections.add(section)

    if _asks_price(
        question,
        language,
    ):
        sections.add("price")

    if _asks_inventory(
        question,
        language,
    ):
        sections.add("inventory")

    if any(term in q for term in ("preserv", "conservateur", "conservation")):
        sections.update({"ingredients", "description", "benefits_and_details"})
    return sections


# ---------------------------------------------------------
# Detect products mentioned in natural-language question
# ---------------------------------------------------------

def find_products_in_question(
    question: str,
) -> List[Dict[str, Any]]:
    """
    Detect product titles, SKUs or barcodes appearing in a
    customer question using the current edsAI catalog.
    """

    q = _normalize(question)

    if not q:
        return []

    matches = []

    for product in _products():
        title = _normalize(
            product.get("title")
        )

        variant_title = _normalize(
            product.get("variantTitle")
        )

        sku = _normalize(
            product.get("sku")
        )

        barcode = _normalize(
            product.get("barcode")
        )

        title_match = (
            title
            and (
                title in q
                or q in title
            )
        )

        sku_match = (
            sku
            and len(sku) >= 4
            and sku in q
        )

        barcode_match = (
            barcode
            and len(barcode) >= 4
            and barcode in q
        )

        # Also support:
        # "Thealoz Duo Gel"
        full_variant_name = _normalize(
            " ".join(
                [
                    str(
                        product.get("title")
                        or ""
                    ),
                    str(
                        product.get("variantTitle")
                        or ""
                    ),
                ]
            )
        )

        variant_match = (
            full_variant_name
            and full_variant_name in q
        )

        if (
            title_match
            or sku_match
            or barcode_match
            or variant_match
        ):
            matches.append(product)

    # Prefer longer / more specific titles first.
    matches.sort(
        key=lambda item: len(
            _normalize(
                item.get("title")
            )
        ),
        reverse=True,
    )

    return matches

NEED_RULES = {
    "redness": {
        "user_terms": {
            "en": (
                "red eye",
                "red eyes",
                "redness",
                "eyes are red",
                "eye is red",
            ),
            "fr": (
                "oeil rouge",
                "yeux rouges",
                "rougeur",
                "rougeurs",
            ),
        },
        "evidence_terms": (
            "redness relief",
            "relief of redness",
            "relieve redness",
            "relieves redness",
            "reduce redness",
            "reduces redness",
            "redness reliever",
            "for red eyes",
        ),
    },

    "dry_eye": {
        "user_terms": {
            "en": (
                "dry eye",
                "dry eyes",
                "dryness",
                "eyes are dry",
                "eyes are very dry",
                "eye is dry",
                "eyes feel dry",
                "eye feels dry",
            ),
            "fr": (
                "oeil sec",
                "yeux secs",
                "secheresse",
                "secheresse oculaire",
            ),
        },
        "evidence_terms": (
            "dry eye",
            "dry eyes",
            "dryness relief",
            "dry eye relief",
            "dry eye symptoms",
            "relief for dry eyes",
        ),
    },

    "allergy": {
        "user_terms": {
            "en": (
                "eye allergy",
                "eye allergies",
                "allergy",
                "allergies",
                "allergic",
                "allergic eyes",
            ),
            "fr": (
                "allergie",
                "allergies",
                "allergique",
                "allergies oculaires",
            ),
        },
        "evidence_terms": (
            "allergy relief",
            "allergy symptoms",
            "eye allergies",
            "allergic conjunctivitis",
            "allergic eyes",
            "allergies",
        ),
    },

    "burning": {
        "user_terms": {
            "en": (
                "burning eye",
                "burning eyes",
                "eyes are burning",
                "eye is burning",
                "burning sensation",
                "burning",
            ),
            "fr": (
                "yeux qui brulent",
                "oeil qui brule",
                "brulure",
                "brulures",
            ),
        },
        "evidence_terms": (
            "relieves burning",
            "relieve burning",
            "burning relief",
            "burning eyes",
            "burning sensation",
            "itching and burning",
        ),
    },

    "itching": {
        "user_terms": {
            "en": (
                "itchy eye",
                "itchy eyes",
                "eye is itchy",
                "eyes are itchy",
                "eye feels itchy",
                "eyes feel itchy",
                "eye itching",
                "itching eye",
                "itching eyes",
                "itching",
                "itchy",
            ),
            "fr": (
                "yeux qui demangent",
                "oeil qui demange",
                "demangeaison",
                "demangeaisons",
            ),
        },
        "evidence_terms": (
            "relieves itching",
            "relieve itching",
            "itching relief",
            "itchy eyes",
            "eye itching",
            "itching and burning",
            "itching",
        ),
    },

    "irritation": {
        "user_terms": {
            "en": (
                "eye irritation",
                "irritated eye",
                "irritated eyes",
                "eyes are irritated",
            ),
            "fr": (
                "irritation oculaire",
                "oeil irrite",
                "yeux irrites",
            ),
        },
        "evidence_terms": (
            "relieves irritation",
            "relieve irritation",
            "irritation relief",
            "eye irritation",
            "irritated eyes",
        ),
    },

    "watery_eyes": {
        "user_terms": {
            "en": (
                "watery eye",
                "watery eyes",
                "eye is watery",
                "eyes are watery",
                "eye feels watery",
                "eyes feel watery",
                "eye is watering",
                "eyes are watering",
                "watering eye",
                "watering eyes",
                "excessive tearing",
                "tearing",
            ),
            "fr": (
                "yeux larmoyants",
                "oeil larmoyant",
                "yeux qui pleurent",
                "oeil qui pleure",
                "larmoiement",
            ),
        },

        "evidence_terms": (
            "watery eye",
            "watery eyes",
            "watering eyes",
            "eyes are watering",
            "excessive tearing",
        ),
    },

    "blepharitis": {
        "user_terms": {
            "en": (
                "blepharitis",
                "inflamed eyelids",
            ),
            "fr": (
                "blepharite",
                "paupiere inflamee",
                "paupières enflammees",
            ),
        },
        "evidence_terms": (
            "blepharitis",
            "blepharitis symptoms",
        ),
    },

    "mgd": {
        "user_terms": {
            "en": (
                "meibomian gland dysfunction",
                "mgd",
                "blocked oil glands",
            ),
            "fr": (
                "dysfonction des glandes de meibomius",
                "glandes de meibomius",
            ),
        },
        "evidence_terms": (
            "meibomian gland dysfunction",
            "meibomian glands",
            "mgd",
        ),
    },

    "contact_lens_discomfort": {
        "user_terms": {
            "en": (
                "contact lens discomfort",
                "contacts feel dry",
                "dry with contacts",
                "contact lenses feel dry",
            ),
            "fr": (
                "lentilles inconfortables",
                "lentilles seches",
            ),
        },
        "evidence_terms": (
            "rewetting drops",
            "rewet contact lenses",
            "contact lens comfort",
            "contact lens dryness",
        ),
    },
}

CONTEXT_GATED_PRODUCTS = {
    "labreleev day for corneal swelling": (
        "corneal swelling",
        "corneal edema",
        "corneal oedema",
    ),
}

def find_products_for_need(
    need: str,
    language: str = "en",
    limit: int = 5,
    product_form: Optional[str] = None,
    question: str = "",
) -> List[Dict[str, Any]]:

    rule = NEED_RULES.get(need)

    normalized_question = _normalize(
        question
    )

    if not rule:
        return []

    evidence_terms = tuple(
        _normalize(term)
        for term in rule["evidence_terms"]
    )

    candidates = []

    for product in _products():
        title = str(
            product.get("title") or ""
        )

        normalized_title = _normalize(
            title
        )

        required_context = (
            CONTEXT_GATED_PRODUCTS.get(
                normalized_title
            )
        )

        context_match = False

        if required_context:
            context_match = any(
                _normalize(term)
                in normalized_question
                for term in required_context
            )

            if not context_match:
                continue

        product_type = str(
            product.get("productType") or ""
        )

        description = str(
            product.get("description") or ""
        )

        metafields = clean_metafields(
            product.get("metafields", [])
        )

        benefits = str(
            metafields.get(
                "accordion_benefits",
                "",
            )
        )

        more_details = str(
            metafields.get(
                "more_details",
                "",
            )
        )

        product_details = str(
            metafields.get(
                "product_details",
                "",
            )
        )

        features = str(
            metafields.get(
                "features",
                "",
            )
        )

        title_text = _normalize(
            title
        )

        description_text = _normalize(
            description
        )

        benefits_text = _normalize(
            benefits
        )

        details_text = _normalize(
            product_details
        )

        features_text = _normalize(
            features
        )

        more_details_text = _normalize(
            more_details
        )

        searchable_text = _normalize(
            " ".join(
                [
                    title,
                    product_type,
                    description,
                    benefits,
                    more_details,
                    product_details,
                    features,
                ]
            )
        )

        # Never return store-only items.
        if any(
            term in title_text
            for term in (
                "gift card",
                "gift certificate",
            )
        ):
            continue

        # If the customer asks for eye drops,
        # exclude wipes, cleansers, supplements, gels, etc.
        if product_form == "eye_drops":
            form_text = _normalize(
                f"{title} {product_type}"
            )

            eye_drop_markers = (
                "eye drop",
                "eye drops",
                "ophthalmic drop",
                "ophthalmic drops",
                "lubricant eye drop",
                "lubricant eye drops",
                "artificial tear",
                "artificial tears",
            )

            if not any(
                _normalize(marker) in form_text
                for marker in eye_drop_markers
            ):
                continue

        # The approved product information must explicitly
        # support the requested need.
        matched_terms = [
            term
            for term in evidence_terms
            if term in searchable_text
        ]

        if not matched_terms:
            continue

        score = 0

        for term in evidence_terms:
            if term in title_text:
                score += 100

            if term in description_text:
                score += 30

            if term in benefits_text:
                score += 25

            if term in details_text:
                score += 20

            if term in features_text:
                score += 20

            if term in more_details_text:
                score += 10

            # Strongly prioritize a condition-specific product
            # when the customer explicitly mentions that condition.
            if context_match:
                score += 250

        candidates.append(
            (
                score,
                product,
            )
        )

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    # Deduplicate Shopify variants only after ranking.
    results = []
    seen = set()

    for _, product in candidates:
        key = (
            product.get("productId")
            or _normalize(
                product.get("title")
            )
        )

        if key in seen:
            continue

        seen.add(key)
        results.append(product)

        if len(results) >= limit:
            break

    return results

def retrieve_products_for_need(
    need: str,
    language: str,
    limit: int = 5,
    product_form: Optional[str] = None,
    question: str = "",
) -> List[Dict[str, Any]]:

    products = find_products_for_need(
        need=need,
        language=language,
        limit=limit,
        product_form=product_form,
        question=question,
    )

    passages = []

    for product in products:
        passages.extend(
            _product_passages(
                product,
                {
                    "description",
                    "benefits_and_details",
                },
                language,
            )
        )

    return passages

def _detect_product_form(
    question: str,
) -> Optional[str]:
    q = _normalize(question)

    if re.search(
        r"\b(?:eye\s+)?drops?\b",
        q,
    ):
        return "eye_drops"

    return None


def retrieve_products_for_needs(
    needs: List[str],
    language: str,
    limit: int = 5,
    product_form: Optional[str] = None,
    question: str = "",
) -> List[Dict[str, Any]]:

    product_matches = {}

    for need in needs:
        products = find_products_for_need(
            need=need,
            language=language,
            limit=limit,
            product_form=product_form,
            question=question,
        )

        for rank, product in enumerate(products):
            key = (
                product.get("productId")
                or _normalize(
                    product.get("title")
                )
            )

            if key not in product_matches:
                product_matches[key] = {
                    "product": product,
                    "needs": set(),
                    "best_rank": rank,
                }

            product_matches[key]["needs"].add(
                need
            )

            product_matches[key]["best_rank"] = min(
                product_matches[key]["best_rank"],
                rank,
            )

    ranked = sorted(
        product_matches.values(),
        key=lambda item: (
            -len(item["needs"]),
            item["best_rank"],
        ),
    )

    selected_products = [
        item["product"]
        for item in ranked[:limit]
    ]

    passages = []

    for product in selected_products:
        passages.extend(
            _product_passages(
                product,
                {
                    "description",
                    "benefits_and_details",
                },
                language,
            )
        )

    return passages
# ---------------------------------------------------------
# Passage creation
# ---------------------------------------------------------

def _passage(
    *,
    variant_id: str,
    product_id: str,
    title: str,
    section: str,
    text: str,
    metadata_extra: Optional[
        Dict[str, Any]
    ] = None,
) -> Dict[str, Any]:

    metadata = {
        "source": "edsAI",
        "product_title": title,
        "section": section,
        "variant_id": variant_id,
        "product_id": product_id,
        "live_api": True,
        "environment": "production",
    }

    if metadata_extra:
        metadata.update(
            metadata_extra
        )

    return {
        "id": (
            f"edsai:{variant_id}:{section}"
        ),
        "text": text,
        "distance": 0.0,
        "metadata": metadata,
    }


def _inventory_lookup():
    return {
        item.get("variantId"): item
        for item in _inventory()
        if item.get("variantId")
    }


def _product_passages(
    product: Dict[str, Any],
    requested_sections: set[str],
    language: str,
) -> List[Dict[str, Any]]:

    variant_id = product.get("id")
    product_id = product.get(
        "productId"
    )
    title = product.get("title") or "Product"

    variant_title = product.get(
        "variantTitle"
    )

    if (
        variant_title
        and variant_title != "Default Title"
    ):
        variant_label = (
            f"{title} — {variant_title}"
        )
    else:
        variant_label = title

    metafields = clean_metafields(
        product.get(
            "metafields",
            [],
        )
    )

    passages = []

    # -----------------------------------------------------
    # Generic description
    # -----------------------------------------------------

    description = product.get(
        "description"
    )

    if (
        not requested_sections
        or "description"
        in requested_sections
    ):
        parts = [
            f"Product: {title}.",
        ]

        variant_title = product.get(
            "variantTitle"
        )

        if (
            variant_title
            and variant_title
            != "Default Title"
        ):
            parts.append(
                f"Variant: {variant_title}."
            )

        vendor = product.get(
            "vendor"
        )

        if vendor:
            parts.append(
                f"Brand/vendor: {vendor}."
            )

        product_type = product.get(
            "productType"
        )

        if product_type:
            parts.append(
                f"Product type: {product_type}."
            )

        if description:
            parts.append(
                f"Description: {description}"
            )

        passages.append(
            _passage(
                variant_id=variant_id,
                product_id=product_id,
                title=title,
                section="description",
                text=" ".join(parts),
            )
        )

    # -----------------------------------------------------
    # Price
    # -----------------------------------------------------

    if "price" in requested_sections:
        retail_price = product.get(
            "retailPrice"
        )

        if retail_price is not None:
            passages.append(
                _passage(
                    variant_id=variant_id,
                    product_id=product_id,
                    title=title,
                    section="price",
                    text=(
                        f"Variant: {variant_label}. "
                        f"Retail price: {retail_price}. "
                        "The edsAI response does not "
                        "provide a currency field."
                    ),
                )
            )

    # -----------------------------------------------------
    # Ingredients
    # -----------------------------------------------------

    if (
        not requested_sections
        or "ingredients"
        in requested_sections
    ):
        ingredients = metafields.get(
            "ingredients"
        )

        if ingredients:
            passages.append(
                _passage(
                    variant_id=variant_id,
                    product_id=product_id,
                    title=title,
                    section="ingredients",
                    text=str(
                        ingredients
                    ),
                )
            )

    # -----------------------------------------------------
    # Usage
    # -----------------------------------------------------

    if (
        not requested_sections
        or "usage"
        in requested_sections
    ):
        usage_values = []

        for key in (
            "how_to_use",
            "dosage",
        ):
            value = metafields.get(
                key
            )

            if value:
                usage_values.append(
                    str(value)
                )

        if usage_values:
            passages.append(
                _passage(
                    variant_id=variant_id,
                    product_id=product_id,
                    title=title,
                    section="usage",
                    text="\n".join(
                        usage_values
                    ),
                )
            )

    # -----------------------------------------------------
    # Warnings
    # -----------------------------------------------------

    if (
        not requested_sections
        or "warnings"
        in requested_sections
    ):
        warnings = metafields.get(
            "warnings"
        )

        if warnings:
            passages.append(
                _passage(
                    variant_id=variant_id,
                    product_id=product_id,
                    title=title,
                    section="warnings",
                    text=str(
                        warnings
                    ),
                )
            )

    # -----------------------------------------------------
    # Benefits / details
    # -----------------------------------------------------

    if (
        not requested_sections
        or "benefits_and_details"
        in requested_sections
    ):
        details = []

        for key in (
            "accordion_benefits",
            "more_details",
            "product_details",
            "features",
            "guarantee",
        ):
            value = metafields.get(
                key
            )

            if value:
                details.append(
                    str(value)
                )

        if details:
            passages.append(
                _passage(
                    variant_id=variant_id,
                    product_id=product_id,
                    title=title,
                    section="benefits_and_details",
                    text="\n".join(
                        details
                    ),
                )
            )

    # -----------------------------------------------------
    # Inventory
    # -----------------------------------------------------

    if "inventory" in requested_sections:
        inventory = _inventory_lookup().get(
            variant_id
        )

        if inventory:
            tracked = inventory.get(
                "tracked"
            )

            available = inventory.get(
                "available"
            )

            if tracked is False or available is None:
                inventory_text = (
                    f"Variant: {variant_label}. "
                    "Inventory is not tracked for this "
                    "variant. Current availability cannot "
                    "be confirmed from edsAI."
                )
            else:
                inventory_text = (
                    f"Variant: {variant_label}. "
                    f"Current aggregate inventory: "
                    f"{available}. "
                    "This is not a reservation or "
                    "delivery guarantee."
                )

            passages.append(
                _passage(
                    variant_id=variant_id,
                    product_id=product_id,
                    title=title,
                    section="inventory",
                    text=inventory_text,
                )
            )

    # -----------------------------------------------------
    # Expiry
    # -----------------------------------------------------

    expiry = metafields.get(
        "expiry_date"
    )

    if expiry and (not requested_sections or "product_details" in requested_sections):
        passages.append(
            _passage(
                variant_id=variant_id,
                product_id=product_id,
                title=title,
                section="product_details",
                text=(
                    f"Variant: {variant_label}. "
                    f"Expiry date: {expiry}."
                ),
                metadata_extra={"variant_title": variant_title},
            )
        )

    # Preserve the commerce identifiers needed by the Shopify storefront.
    # These values come directly from edsAI and remain structured metadata;
    # the language model never creates or alters them.
    for passage in passages:
        passage["metadata"].update({
            "variant_title": variant_title,
            "image_url": product.get("image"),
        })

    return passages


# ---------------------------------------------------------
# Public retrieval function
# ---------------------------------------------------------
def _detect_product_needs(
    question: str,
    language: str,
) -> List[str]:

    q = _normalize(
        question
    )

    matches = []

    language_key = (
        "fr"
        if language == "fr"
        else "en"
    )

    for need, rule in NEED_RULES.items():
        terms = rule[
            "user_terms"
        ].get(
            language_key,
            (),
        )

        if any(
            _normalize(term) in q
            for term in terms
        ):
            matches.append(
                need
            )

    return matches
def retrieve_edsai_passages(
    question: str,
    language: str,
) -> List[Dict[str, Any]]:

    current_question = _current_question_text(
        question
    )

    understanding = understand_query(
        current_question,
        language,
    )

    print(
        "[edsAI-understanding]",
        understanding,
    )

    # ---------------------------------------------
    # Safety routing
    # ---------------------------------------------

    if (
        understanding.get(
            "personal_symptom"
        )
        and understanding.get(
            "high_risk"
        )
    ):
        return []

    # ---------------------------------------------
    # Explicit named product
    #
    # Preserve your deterministic named-product
    # path for prices, inventory, usage, etc.
    # ---------------------------------------------

    named_product_query = (
        understanding.get(
            "named_product_query"
        )
    )

    products = []

    # Only run deterministic product-name matching when
    # the LLM says the user actually mentioned a product.
    if named_product_query:
        products = find_products_in_question(
            named_product_query
        )

        # Prefer an exact product-title match.
        # Example:
        # "HYLO-DUAL" should match HYLO-DUAL®,
        # not the broader HYLO® product.
        normalized_named = _normalize(
            named_product_query
        )

        exact_products = [
            product
            for product in products
            if _normalize(
                product.get("title")
            )
            == normalized_named
        ]

        if exact_products:
            products = exact_products


    # Preserve every explicitly named item when comparing, even if query understanding
    # emitted only one product name.
    comparison = any(t in _normalize(current_question) for t in
                     ('compare', 'comparaison', 'difference', 'versus', ' vs ', 'cheaper', 'moins cher'))
    if comparison:
        products = find_products_in_question(current_question)
    if products and understanding.get("intent") != "product_search":
        # The semantic query plan selects evidence fields. Keep the legacy
        # selector only for callers supplying an older plan without this field.
        planned_sections = understanding.get("requested_sections")
        requested_sections = (
            set(planned_sections) if planned_sections is not None
            else _requested_sections(current_question, language)
        )
        if not requested_sections:
            requested_sections = {"description"}

        if comparison:
            requested_sections |= {"description", "ingredients", "benefits_and_details", "price", "inventory"}
        passages = []

        for product in products[:6]:
            passages.extend(
                _product_passages(
                    product,
                    requested_sections,
                    language,
                )
            )

        return passages

    # ---------------------------------------------
    # LLM-understood symptom / need discovery
    # ---------------------------------------------

    if understanding.get("intent") not in {
        "product_search",
        "symptom_discovery",
        "symptom_statement",
    }:
        return []

    catalog = _products()

    candidates = semantic_candidates(
        search_query=(
            understanding.get(
                "search_query"
            )
            or current_question
        ),
        products=catalog,
        catalog_version=_product_cache_time,
        top_k=8,
        requested_form=(
            understanding.get(
                "requested_form"
            )
            or "any"
        ),
    )

    selected_products = rerank_candidates(
        question=current_question,
        understanding=understanding,
        candidates=candidates,
        max_results=5,
    )

    passages = []

    for product in selected_products:
        product_passages = _product_passages(
            product,
            {
                "description",
                "benefits_and_details",
            },
            language,
        )

        for passage in product_passages:
            passage.setdefault(
                "metadata",
                {},
            )

            passage["metadata"][
                "discovery_label"
            ] = understanding.get(
                "response_label"
            )

            passage["metadata"][
                "query_intent"
            ] = understanding.get(
                "intent"
            )

        passages.extend(
            product_passages
        )

    return passages
PRODUCT_SECTIONS = {
    "description",
    "ingredients",
    "warnings",
    "usage",
    "benefits_and_details",
    "product_details",
    "price",
    "inventory",
    "availability",
}

def merge_with_existing_passages(
    existing_passages,
    edsai_passages,
):
    """
    When edsAI identifies a real product, edsAI becomes
    authoritative for Shopify product data.

    Keep non-product knowledge such as shipping/return policy.
    """

    if not edsai_passages:
        return existing_passages

    filtered = []

    for passage in existing_passages or []:
        metadata = passage.get(
            "metadata",
            {},
        )

        section = metadata.get(
            "section"
        )

        appears_product_specific = any(
            metadata.get(key)
            for key in (
                "product_title",
                "product_handle",
                "product_id",
                "variant_id",
                "sku",
            )
        )

        if (
            appears_product_specific
            and section in PRODUCT_SECTIONS
        ):
            # Drop old/static product evidence.
            continue

        filtered.append(
            passage
        )

    # Put live evidence first so its source numbers are
    # stable and obvious.
    return (
        edsai_passages
        + filtered
    )
