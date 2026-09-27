from __future__ import annotations

import os
from functools import lru_cache

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from product_matcher import (
    detect_exact_product,
    detect_product_mentions,
)
from retrieval import (
    EMBEDDING_MODEL,
    MAX_DISTANCE,
    NUMBER_OF_RESULTS,
    POLICY_SECTIONS,
    PRODUCT_SECTIONS,
    expand_educational_query,
    _finalize_candidates,
    _with_priority,
    detect_section_intent,
)


POSTGRES_DSN = os.getenv(
    "BRAIN_POSTGRES_DSN",
    "dbname=eyedropshop_brain",
)

VALID_AUDIENCES = {
    "internal",
    "doctor",
    "customer",
}


@lru_cache(maxsize=1)
def load_embedding_model():
    return SentenceTransformer(
        EMBEDDING_MODEL
    )


def _validate_audience(audience):
    if audience not in VALID_AUDIENCES:
        raise ValueError(
            f"Invalid audience: {audience}"
        )


def _where_to_sql(where):
    """
    Translate the small subset of Chroma filters
    used by the current retrieval router into SQL.
    """
    if where is None:
        return "", []

    if "$and" in where:
        clauses = []
        params = []

        for condition in where["$and"]:
            clause, values = _where_to_sql(
                condition
            )

            if clause:
                clauses.append(
                    f"({clause})"
                )
                params.extend(values)

        return (
            " AND ".join(clauses),
            params,
        )

    allowed_fields = {
        "language": (
            "p.metadata->>'language'"
        ),
        "product_handle": (
            "p.metadata->>'product_handle'"
        ),
        "section": (
            "p.metadata->>'section'"
        ),
    }

    clauses = []
    params = []

    for key, value in where.items():
        column = allowed_fields.get(key)

        if column is None:
            raise ValueError(
                "Unsupported PostgreSQL "
                f"retrieval filter: {key}"
            )

        if isinstance(value, dict):
            if "$in" not in value:
                raise ValueError(
                    "Unsupported filter operator "
                    f"for {key}: {value}"
                )

            clauses.append(
                f"{column} = ANY(%s::text[])"
            )
            params.append(
                list(value["$in"])
            )

        else:
            clauses.append(
                f"{column} = %s"
            )
            params.append(value)

    return (
        " AND ".join(clauses),
        params,
    )


def _query_postgres(
    question_embedding,
    where,
    n_results,
    audience="internal",
):
    _validate_audience(audience)

    allowed_audiences = [
        audience,
        "shared",
    ]

    filter_sql, filter_params = (
        _where_to_sql(where)
    )

    clauses = [
        "s.approval_status = 'approved'",
        (
            "s.audience = "
            "ANY(%s::text[])"
        ),
        (
            "("
            "s.expires_at IS NULL "
            "OR s.expires_at >= CURRENT_DATE"
            ")"
        ),
    ]

    params = [
        allowed_audiences,
    ]

    if filter_sql:
        clauses.append(
            f"({filter_sql})"
        )
        params.extend(
            filter_params
        )

    where_sql = " AND ".join(
        clauses
    )

    vector = Vector(
        [
            float(value)
            for value
            in question_embedding[0]
        ]
    )

    # Chroma's current default L2 distance
    # on normalized embeddings is squared L2.
    #
    # pgvector's <-> returns L2 distance,
    # so squaring it makes the distance
    # compatible with the current Brain's
    # MAX_DISTANCE threshold.
    sql = f"""
        SELECT
            p.passage_id,
            p.content,
            p.metadata,
            POWER(
                p.embedding <-> %s,
                2
            ) AS distance
        FROM knowledge_passages p
        JOIN knowledge_sources s
            ON s.registry_id =
               p.source_registry_id
        WHERE {where_sql}
        ORDER BY
            p.embedding <-> %s
        LIMIT %s
    """

    query_params = [
        vector,
        *params,
        vector,
        n_results,
    ]

    passages = []

    with psycopg.connect(
        POSTGRES_DSN
    ) as conn:
        register_vector(conn)

        rows = conn.execute(
            sql,
            query_params,
        ).fetchall()

    for (
        passage_id,
        content,
        metadata,
        distance,
    ) in rows:
        passages.append(
            {
                "id": passage_id,
                "text": content,
                "metadata": metadata,
                "distance": float(
                    distance
                ),
            }
        )

    return passages


def retrieve_passages_postgres(
    question: str,
    language: str,
    number_of_results: int = NUMBER_OF_RESULTS,
    max_distance: float = MAX_DISTANCE,
    audience: str = "internal",
    embedding_model=None,
):
    """
    PostgreSQL/pgvector implementation
    of the current Brain retrieval logic.
    """
    _validate_audience(
        audience
    )

    if embedding_model is None:
        embedding_model = (
            load_embedding_model()
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

    # ----------------------------------------------------------
    # A) Multiple explicitly named products
    # ----------------------------------------------------------
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
                    _query_postgres(
                        question_embedding,
                        {
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
                        1,
                        audience,
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
                _query_postgres(
                    question_embedding,
                    product_filter,
                    per_product,
                    audience,
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

    # ----------------------------------------------------------
    # B) One explicitly named product
    # ----------------------------------------------------------
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
                _query_postgres(
                    question_embedding,
                    {
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
                    1,
                    audience,
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
                _query_postgres(
                    question_embedding,
                    product_filter,
                    number_of_results,
                    audience,
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
                _query_postgres(
                    question_embedding,
                    {
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
                    number_of_results,
                    audience,
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
                _query_postgres(
                    question_embedding,
                    product_filter,
                    1,
                    audience,
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
                _query_postgres(
                    question_embedding,
                    product_filter,
                    number_of_results,
                    audience,
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

    else:
        # ------------------------------------------------------
        # No product explicitly named.
        #
        # Policy intent may safely constrain retrieval because
        # shipping/returns refer to dedicated policy sources.
        #
        # Product-section words such as "ingredients", "dosage",
        # "warnings", or "features" must NOT restrict retrieval
        # when no product is named. Those words can also occur in
        # educational transcript questions.
        # ------------------------------------------------------
        if (
            section_intent
            in POLICY_SECTIONS
        ):
            policy_passages = (
                _query_postgres(
                    question_embedding,
                    {
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
                    number_of_results,
                    audience,
                )
            )

            candidates.extend(
                _with_priority(
                    policy_passages,
                    priority=0,
                    force_include=True,
                )
            )

        else:
            global_candidates = (
                _query_postgres(
                    question_embedding,
                    {
                        "language":
                        language
                    },
                    number_of_results,
                    audience,
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

    # ----------------------------------------------------------
    # Education-only fallback
    #
    # If normal retrieval found nothing, no product was named,
    # and this is not a policy question, try approved educational
    # transcript passages with a slightly looser threshold.
    # ----------------------------------------------------------
    if (
        not passages
        and matched_product is None
        and section_intent not in POLICY_SECTIONS
    ):
        education_candidates = (
            _query_postgres(
                question_embedding,
                {
                    "$and": [
                        {
                            "language":
                            language
                        },
                        {
                            "section":
                            "education"
                        },
                    ]
                },
                number_of_results,
                audience,
            )
        )

        passages = _finalize_candidates(
            candidates=_with_priority(
                education_candidates,
                priority=1,
            ),
            number_of_results=number_of_results,
            max_distance=1.10,
        )

    return passages, matched_product
