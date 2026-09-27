from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from product_matcher import (
    detect_exact_product,
    detect_product_mentions,
    normalize_text,
)


BASE_DIR = Path(__file__).resolve().parent
DATABASE_FOLDER = BASE_DIR / "chroma_products_db"
COLLECTION_NAME = "eyedropshop_products"
EMBEDDING_MODEL = "BAAI/bge-m3"
NUMBER_OF_RESULTS = 5
MAX_DISTANCE = 0.85


SECTION_KEYWORDS = {
    "en": {
        "returns": [
            "return",
            "returns",
            "return policy",
            "refund",
            "refunds",
            "refunded",
            "refunding",
            "exchange",
            "exchanges",
        ],
        "shipping": [
            "shipping",
            "ship",
            "ships",
            "delivery",
            "deliver",
            "delivered",
        ],
        "ingredients": [
            "ingredient",
            "ingredients",
            "active ingredient",
            "inactive ingredient",
            "contains",
            "contain",
            "made of",
            "preservative",
            "preservatives",
            "preservative free",
        ],
        "warnings": [
            "warning",
            "warnings",
            "caution",
            "precaution",
            "precautions",
            "avoid",
            "contraindication",
            "contraindications",
        ],
        "usage": [
            "how do i use",
            "how to use",
            "directions",
            "direction for use",
            "dosage",
            "dose",
            "apply",
            "application",
            "instructions",
            "spray",
            "spraying",
            "contact lens",
            "contact lenses",
        ],
        "benefits_and_details": [
            "benefit",
            "benefits",
            "feature",
            "features",
            "advantages",
        ],
    },
    "fr": {
        "returns": [
            "politique de retour",
            "retour",
            "retours",
            "retourner",
            "remboursement",
            "rembourser",
            "rembourse",
            "rembourses",
            "remboursee",
            "remboursees",
            "echange",
            "echanges",
        ],
        "shipping": [
            "livraison",
            "livrer",
            "expedition",
            "expedier",
            "delai de livraison",
            "delais de livraison",
        ],
        "ingredients": [
            "ingredient",
            "ingredients",
            "ingredient actif",
            "ingredients actifs",
            "contient",
            "composition",
            "agent de conservation",
            "agents de conservation",
            "sans agent de conservation",
            "sans conservateur",
        ],
        "warnings": [
            "avertissement",
            "avertissements",
            "attention",
            "precaution",
            "precautions",
            "mise en garde",
            "mises en garde",
            "eviter",
            "contre indication",
            "contre indications",
        ],
        "usage": [
            "comment utiliser",
            "mode d emploi",
            "mode d utilisation",
            "utilisation",
            "dosage",
            "dose",
            "appliquer",
            "application",
            "instructions",
            "vaporiser",
            "vaporisez",
            "vaporisation",
            "lentille de contact",
            "lentilles de contact",
        ],
        "benefits_and_details": [
            "avantage",
            "avantages",
            "benefice",
            "benefices",
            "caracteristique",
            "caracteristiques",
        ],
    },
}


PRODUCT_SECTIONS = {
    "description",
    "benefits_and_details",
    "usage",
    "ingredients",
    "warnings",
}

POLICY_SECTIONS = {
    "shipping",
    "returns",
}


VALID_AUDIENCES = {
    "internal",
    "doctor",
    "customer",
}


@lru_cache(maxsize=1)
def load_resources():
    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    client = chromadb.PersistentClient(
        path=str(DATABASE_FOLDER)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    return embedding_model, collection


def detect_section_intent(
    question: str,
    language: str,
):
    """
    Detect which knowledge section the user is asking about.

    Product names are removed before intent detection so words
    appearing inside a product title (for example,
    "Preservative-Free") do not incorrectly trigger a section
    such as "ingredients".
    """

    if language not in SECTION_KEYWORDS:
        return None

    normalized = normalize_text(question)

    # ----------------------------------------------------------
    # Remove explicitly detected product names/aliases before
    # looking for section keywords.
    #
    # Example:
    # "What are the warnings for Systane Hydration
    #  Preservative-Free?"
    #
    # should detect "warnings", not "ingredients".
    # ----------------------------------------------------------
    mentioned_products = detect_product_mentions(
        question,
        language,
    )

    intent_text = normalized

    for product in mentioned_products:
        product_terms = [
            product.matched_alias,
            product.product_title,
        ]

        for term in product_terms:
            if not term:
                continue

            normalized_term = normalize_text(
                term
            )

            if normalized_term:
                intent_text = intent_text.replace(
                    normalized_term,
                    " ",
                )

    # Normalize whitespace after removing product names.
    intent_text = " ".join(
        intent_text.split()
    )

    # ----------------------------------------------------------
    # Detect section intent from the remaining question text.
    # ----------------------------------------------------------
    for section, keywords in (
        SECTION_KEYWORDS[language].items()
    ):
        for keyword in keywords:
            normalized_keyword = normalize_text(
                keyword
            )

            if (
                normalized_keyword
                and normalized_keyword
                in intent_text
            ):
                return section

    return None


def _allowed_knowledge_audiences(
    audience,
):
    """
    Return the knowledge audiences that a requester
    is permitted to retrieve.

    Every requester can access:
    - their own audience-specific content;
    - shared content.
    """
    if audience not in VALID_AUDIENCES:
        raise ValueError(
            f"Invalid audience: {audience}"
        )

    return {
        audience,
        "shared",
    }


def _governed_where(
    where,
    audience,
):
    """
    Enforce approval and audience permissions
    at the Chroma query level.
    """
    allowed_audiences = list(
        _allowed_knowledge_audiences(
            audience
        )
    )

    governance_filters = [
        {
            "approval_status": "approved",
        },
        {
            "audience": {
                "$in": allowed_audiences,
            }
        },
    ]

    if where is None:
        return {
            "$and": governance_filters,
        }

    if "$and" in where:
        return {
            "$and": [
                *where["$and"],
                *governance_filters,
            ]
        }

    return {
        "$and": [
            where,
            *governance_filters,
        ]
    }


def _metadata_is_retrievable(
    metadata,
    audience,
):
    """
    Final fail-closed governance check.

    A record must:
    - be approved;
    - belong to the requesting audience or shared;
    - not be expired.

    Invalid expiration metadata is rejected.
    """
    if (
        metadata.get("approval_status")
        != "approved"
    ):
        return False

    allowed_audiences = (
        _allowed_knowledge_audiences(
            audience
        )
    )

    if (
        metadata.get("audience")
        not in allowed_audiences
    ):
        return False

    expires_at = str(
        metadata.get(
            "expires_at",
            "",
        )
    ).strip()

    if not expires_at:
        return True

    try:
        expiration_date = (
            date.fromisoformat(
                expires_at[:10]
            )
        )
    except ValueError:
        return False

    return (
        expiration_date
        >= date.today()
    )


def _query_collection(
    question_embedding,
    where,
    n_results,
    audience="internal",
):
    _, collection = load_resources()

    governed_where = _governed_where(
        where,
        audience,
    )

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=n_results,
        where=governed_where,
    )

    passages = []

    for (
        document_id,
        document,
        metadata,
        distance,
    ) in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        if not _metadata_is_retrievable(
            metadata,
            audience,
        ):
            continue

        passages.append(
            {
                "id": document_id,
                "text": document,
                "metadata": metadata,
                "distance": float(distance),
            }
        )

    return passages


def _with_priority(
    passages,
    priority,
    force_include=False,
):
    prioritized = []

    for passage in passages:
        item = dict(passage)
        item["_priority"] = priority
        item["_force_include"] = (
            force_include
        )

        prioritized.append(item)

    return prioritized


def _finalize_candidates(
    candidates,
    number_of_results,
    max_distance,
):
    unique_by_id = {}

    for passage in candidates:
        if not passage.get(
            "_force_include",
            False,
        ):
            if (
                passage["distance"]
                > max_distance
            ):
                continue

        current = unique_by_id.get(
            passage["id"]
        )

        if current is None:
            unique_by_id[
                passage["id"]
            ] = passage
            continue

        candidate_key = (
            passage.get(
                "_priority",
                99,
            ),
            passage["distance"],
        )

        current_key = (
            current.get(
                "_priority",
                99,
            ),
            current["distance"],
        )

        if candidate_key < current_key:
            unique_by_id[
                passage["id"]
            ] = passage

    passages = sorted(
        unique_by_id.values(),
        key=lambda passage: (
            passage.get(
                "_priority",
                99,
            ),
            passage["distance"],
        ),
    )[:number_of_results]

    for passage in passages:
        passage.pop(
            "_priority",
            None,
        )

        passage.pop(
            "_force_include",
            None,
        )

    return passages

def expand_educational_query(
    question: str,
    language: str,
):
    """
    Add conservative aliases for educational terms and common
    paraphrases used in the approved educational sources.

    The original user question is preserved; expansions are used
    only for retrieval embedding.
    """
    if language != "en":
        return question

    normalized = question.lower()

    expansions = []

    # ----------------------------------------------------------
    # Meibomian gland transcript aliases / mistranscriptions.
    # ----------------------------------------------------------
    if "meibomian" in normalized:
        expansions.extend(
            [
                "mayomian glands",
                "mayabomian glands",
                "oil glands in the eyelids",
            ]
        )

    # ----------------------------------------------------------
    # Tear-film composition paraphrases.
    #
    # Example:
    # "What is tear film made of?"
    # should retrieve passages describing the three layers.
    # ----------------------------------------------------------
    if (
        "tear film" in normalized
        and any(
            phrase in normalized
            for phrase in (
                "made of",
                "composed of",
                "consist of",
                "consists of",
                "composition",
            )
        )
    ):
        expansions.extend(
            [
                "tear film three layers",
                "mucin layer aqueous layer lipid layer",
            ]
        )

    if not expansions:
        return question

    return (
        question
        + " "
        + " ".join(expansions)
    )

def retrieve_passages(
    question: str,
    language: str,
    number_of_results: int = NUMBER_OF_RESULTS,
    max_distance: float = MAX_DISTANCE,
    audience: str = "internal",
):
    """
    Structured + semantic retrieval.

    Handles:
    - one specifically named product;
    - multiple specifically named products for comparison;
    - section-aware product retrieval;
    - shipping and return policies;
    - semantic fallback;
    - audience-aware permission filtering.
    """

    # Validate audience before any retrieval begins.
    _allowed_knowledge_audiences(
        audience
    )

    embedding_model, collection = (
        load_resources()
    )

    retrieval_question = (
        expand_educational_query(
            question,
            language,
        )
    )

    question_embedding = (
        embedding_model.encode(
            [retrieval_question],
            normalize_embeddings=True,
        ).tolist()
    )

    mentioned_products = (
        detect_product_mentions(
            question,
            language,
        )
    )

    matched_product = (
        detect_exact_product(
            question,
            language,
        )
    )

    section_intent = (
        detect_section_intent(
            question,
            language,
        )
    )

    candidates = []

    # --------------------------------------------------------------
    # A) Multiple explicitly named products.
    # --------------------------------------------------------------
    if len(mentioned_products) >= 2:
        per_product = max(
            1,
            number_of_results
            // len(mentioned_products),
        )

        for product in mentioned_products:
            product_filter = {
                "$and": [
                    {
                        "language":
                        language
                    },
                    {
                        "product_handle":
                        product.product_handle
                    },
                ]
            }

            if (
                section_intent
                in PRODUCT_SECTIONS
            ):
                exact_section = (
                    _query_collection(
                        question_embedding=question_embedding,
                        where={
                            "$and": [
                                {
                                    "language":
                                    language
                                },
                                {
                                    "product_handle":
                                    product.product_handle
                                },
                                {
                                    "section":
                                    section_intent
                                },
                            ]
                        },
                        n_results=1,
                        audience=audience,
                    )
                )

                candidates.extend(
                    _with_priority(
                        exact_section,
                        priority=0,
                        force_include=True,
                    )
                )

            product_passages = (
                _query_collection(
                    question_embedding=question_embedding,
                    where=product_filter,
                    n_results=per_product,
                    audience=audience,
                )
            )

            if product_passages:
                candidates.extend(
                    _with_priority(
                        product_passages[:1],
                        priority=1,
                        force_include=True,
                    )
                )

                candidates.extend(
                    _with_priority(
                        product_passages[1:],
                        priority=2,
                    )
                )

    # --------------------------------------------------------------
    # B) One explicitly named product.
    # --------------------------------------------------------------
    elif matched_product is not None:
        product_filter = {
            "$and": [
                {
                    "language":
                    language
                },
                {
                    "product_handle":
                    matched_product.product_handle
                },
            ]
        }

        if (
            section_intent
            in PRODUCT_SECTIONS
        ):
            exact_section = (
                _query_collection(
                    question_embedding=question_embedding,
                    where={
                        "$and": [
                            {
                                "language":
                                language
                            },
                            {
                                "product_handle":
                                matched_product.product_handle
                            },
                            {
                                "section":
                                section_intent
                            },
                        ]
                    },
                    n_results=1,
                    audience=audience,
                )
            )

            candidates.extend(
                _with_priority(
                    exact_section,
                    priority=0,
                    force_include=True,
                )
            )

            product_fallback = (
                _query_collection(
                    question_embedding=question_embedding,
                    where=product_filter,
                    n_results=number_of_results,
                    audience=audience,
                )
            )

            candidates.extend(
                _with_priority(
                    product_fallback,
                    priority=2,
                )
            )

        elif (
            section_intent
            in POLICY_SECTIONS
        ):
            policy_passages = (
                _query_collection(
                    question_embedding=question_embedding,
                    where={
                        "$and": [
                            {
                                "language":
                                language
                            },
                            {
                                "section":
                                section_intent
                            },
                        ]
                    },
                    n_results=number_of_results,
                    audience=audience,
                )
            )

            candidates.extend(
                _with_priority(
                    policy_passages,
                    priority=0,
                    force_include=True,
                )
            )

            product_context = (
                _query_collection(
                    question_embedding=question_embedding,
                    where=product_filter,
                    n_results=1,
                    audience=audience,
                )
            )

            candidates.extend(
                _with_priority(
                    product_context,
                    priority=1,
                    force_include=True,
                )
            )

        else:
            product_candidates = (
                _query_collection(
                    question_embedding=question_embedding,
                    where=product_filter,
                    n_results=number_of_results,
                    audience=audience,
                )
            )

            if product_candidates:
                candidates.extend(
                    _with_priority(
                        product_candidates[:1],
                        priority=0,
                        force_include=True,
                    )
                )

                candidates.extend(
                    _with_priority(
                        product_candidates[1:],
                        priority=2,
                    )
                )

    # --------------------------------------------------------------
    # C) No specific product named.
    # --------------------------------------------------------------
    else:
        if (
            section_intent
            in POLICY_SECTIONS
        ):
            policy_passages = (
                _query_collection(
                    question_embedding=question_embedding,
                    where={
                        "$and": [
                            {
                                "language":
                                language
                            },
                            {
                                "section":
                                section_intent
                            },
                        ]
                    },
                    n_results=number_of_results,
                    audience=audience,
                )
            )

            candidates.extend(
                _with_priority(
                    policy_passages,
                    priority=0,
                    force_include=True,
                )
            )

        elif (
            section_intent
            in PRODUCT_SECTIONS
        ):
            section_candidates = (
                _query_collection(
                    question_embedding=question_embedding,
                    where={
                        "$and": [
                            {
                                "language":
                                language
                            },
                            {
                                "section":
                                section_intent
                            },
                        ]
                    },
                    n_results=number_of_results,
                    audience=audience,
                )
            )

            candidates.extend(
                _with_priority(
                    section_candidates,
                    priority=1,
                )
            )

        else:
            global_candidates = (
                _query_collection(
                    question_embedding=question_embedding,
                    where={
                        "language":
                        language
                    },
                    n_results=min(
                        number_of_results,
                        collection.count(),
                    ),
                    audience=audience,
                )
            )

            candidates.extend(
                _with_priority(
                    global_candidates,
                    priority=1,
                )
            )

    passages = _finalize_candidates(
        candidates=candidates,
        number_of_results=number_of_results,
        max_distance=max_distance,
    )

    return passages, matched_product