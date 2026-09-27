from __future__ import annotations

import argparse

import pandas as pd
import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from governance_registry import (
    get_governance_record,
    load_registry,
)
from retrieval import load_resources


DATABASE_NAME = "eyedropshop_brain"

PRODUCT_FILE = (
    "prototype_data/products/"
    "products_30_bilingual.csv"
)


def clean_value(value):
    if pd.isna(value):
        return ""

    return str(value).strip()


def nullable(value):
    value = str(value or "").strip()

    return value or None


def database_counts(conn):
    counts = {}

    for table in (
        "knowledge_sources",
        "products",
        "knowledge_passages",
    ):
        row = conn.execute(
            f"SELECT COUNT(*) FROM {table}"
        ).fetchone()

        counts[table] = row[0]

    return counts


def migrate_sources(conn):
    rows = load_registry()

    for row in rows:
        conn.execute(
            """
            INSERT INTO knowledge_sources (
                registry_id,
                source_name,
                content_type,
                language,
                country,
                audience,
                approval_status,
                approver,
                knowledge_version,
                review_date,
                expires_at,
                notes
            )
            VALUES (
                %(registry_id)s,
                %(source_name)s,
                %(content_type)s,
                %(language)s,
                %(country)s,
                %(audience)s,
                %(approval_status)s,
                %(approver)s,
                %(knowledge_version)s,
                %(review_date)s,
                %(expires_at)s,
                %(notes)s
            )
            """,
            {
                "registry_id": (
                    row["registry_id"]
                ),
                "source_name": (
                    row["source"]
                ),
                "content_type": (
                    row["content_type"]
                ),
                "language": (
                    row["language"]
                ),
                "country": (
                    row["country"]
                    or "CA"
                ),
                "audience": (
                    row["audience"]
                ),
                "approval_status": (
                    row[
                        "approval_status"
                    ]
                ),
                "approver": nullable(
                    row["approver"]
                ),
                "knowledge_version": (
                    row[
                        "knowledge_version"
                    ]
                ),
                "review_date": nullable(
                    row["review_date"]
                ),
                "expires_at": nullable(
                    row["expires_at"]
                ),
                "notes": nullable(
                    row["notes"]
                ),
            },
        )

    return len(rows)


def migrate_products(conn):
    products = pd.read_csv(
        PRODUCT_FILE
    )

    inserted = 0

    for _, row in products.iterrows():
        source = clean_value(
            row["source"]
        )

        language = clean_value(
            row["language"]
        )

        governance = (
            get_governance_record(
                source=source,
                content_type="product",
                language=language,
                country="CA",
            )
        )

        title = clean_value(
            row["title"]
        )

        url = clean_value(
            row.get("url")
        )

        conn.execute(
            """
            INSERT INTO products (
                product_id,
                language,
                product_handle,
                title,
                country,
                source_registry_id,
                url
            )
            VALUES (
                %(product_id)s,
                %(language)s,
                %(product_handle)s,
                %(title)s,
                %(country)s,
                %(source_registry_id)s,
                %(url)s
            )
            """,
            {
                "product_id": clean_value(
                    row["product_id"]
                ),
                "language": language,
                "product_handle": (
                    clean_value(
                        row[
                            "product_handle"
                        ]
                    )
                ),
                "title": (
                    title
                    if title
                    else None
                ),
                "country": "CA",
                "source_registry_id": (
                    governance[
                        "registry_id"
                    ]
                ),
                "url": (
                    url
                    if url
                    else None
                ),
            },
        )

        inserted += 1

    return inserted


def migrate_passages(conn):
    _, collection = load_resources()

    result = collection.get(
        include=[
            "documents",
            "metadatas",
            "embeddings",
        ]
    )

    ids = result.get(
        "ids",
        []
    ) or []

    documents = result.get(
        "documents",
        []
    ) or []

    metadatas = result.get(
        "metadatas",
        []
    ) or []

    embeddings = result.get(
        "embeddings",
        []
    )

    if embeddings is None:
        raise RuntimeError(
            "Chroma did not return embeddings."
        )

    if not (
        len(ids)
        == len(documents)
        == len(metadatas)
        == len(embeddings)
    ):
        raise RuntimeError(
            "Chroma result lengths do not match."
        )

    inserted = 0

    for (
        passage_id,
        content,
        metadata,
        embedding,
    ) in zip(
        ids,
        documents,
        metadatas,
        embeddings,
    ):
        content_type = str(
            metadata.get(
                "content_type",
                "",
            )
        ).strip()

        source_name = str(
            metadata.get(
                "source",
                "",
            )
        ).strip()

        language = str(
            metadata.get(
                "language",
                "",
            )
        ).strip()

        country = str(
            metadata.get(
                "country",
                "CA",
            )
        ).strip() or "CA"

        governance = get_governance_record(
            source=source_name,
            content_type=content_type,
            language=language,
            country=country,
        )

        for field in (
            "audience",
            "approval_status",
            "approver",
            "knowledge_version",
            "review_date",
            "expires_at",
        ):
            metadata[field] = governance[field]

        registry_id = governance[
            "registry_id"
        ]

        metadata["registry_id"] = registry_id

        if content_type == "product":
            product_id = str(
                metadata.get(
                    "product_id",
                    "",
                )
            ).strip()

            product_language = str(
                metadata.get(
                    "language",
                    "",
                )
            ).strip()

            if (
                not product_id
                or not product_language
            ):
                raise ValueError(
                    f"{passage_id}: "
                    "product passage is "
                    "missing product identity"
                )

        else:
            product_id = None
            product_language = None

        vector = Vector(
            [
                float(value)
                for value in embedding
            ]
        )

        conn.execute(
            """
            INSERT INTO knowledge_passages (
                passage_id,
                source_registry_id,
                product_id,
                product_language,
                section,
                content,
                metadata,
                embedding
            )
            VALUES (
                %(passage_id)s,
                %(source_registry_id)s,
                %(product_id)s,
                %(product_language)s,
                %(section)s,
                %(content)s,
                %(metadata)s,
                %(embedding)s
            )
            """,
            {
                "passage_id": (
                    passage_id
                ),
                "source_registry_id": (
                    registry_id
                ),
                "product_id": (
                    product_id
                ),
                "product_language": (
                    product_language
                ),
                "section": (
                    metadata.get(
                        "section"
                    )
                ),
                "content": content,
                "metadata": Jsonb(
                    metadata
                ),
                "embedding": vector,
            },
        )

        inserted += 1

    return inserted


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--replace",
        action="store_true",
        help=(
            "Clear the PostgreSQL Brain "
            "tables before migration."
        ),
    )

    args = parser.parse_args()

    with psycopg.connect(
        dbname=DATABASE_NAME
    ) as conn:
        register_vector(conn)

        before = database_counts(
            conn
        )

        print(
            "Current PostgreSQL counts:",
            before,
        )

        if any(before.values()):
            if not args.replace:
                raise SystemExit(
                    "PostgreSQL Brain tables "
                    "already contain data.\n"
                    "Use --replace only if you "
                    "intend to rebuild them."
                )

            print(
                "Clearing existing "
                "PostgreSQL Brain data..."
            )

            conn.execute(
                """
                TRUNCATE TABLE
                    knowledge_passages,
                    products,
                    knowledge_sources
                RESTART IDENTITY
                CASCADE
                """
            )

        print(
            "\nMigrating governance "
            "sources..."
        )

        source_count = (
            migrate_sources(conn)
        )

        print(
            "Governance sources:",
            source_count,
        )

        print(
            "\nMigrating products..."
        )

        product_count = (
            migrate_products(conn)
        )

        print(
            "Product rows:",
            product_count,
        )

        print(
            "\nMigrating passages "
            "and embeddings..."
        )

        passage_count = (
            migrate_passages(conn)
        )

        print(
            "Knowledge passages:",
            passage_count,
        )

        after = database_counts(
            conn
        )

        print(
            "\nPostgreSQL counts:",
            after,
        )

        if source_count != 29:
            raise RuntimeError(
                "Expected 29 governance "
                "source records."
            )

        if product_count != 60:
            raise RuntimeError(
                "Expected 60 bilingual "
                "product rows."
            )

        if passage_count != 549:
            raise RuntimeError(
                "Expected 549 passages."
            )

        if after != {
            "knowledge_sources": 29,
            "products": 60,
            "knowledge_passages": 549,
        }:
            raise RuntimeError(
                "Unexpected PostgreSQL counts."
            )
            raise RuntimeError(
                "PostgreSQL final counts "
                "do not match expected data."
            )

        print(
            "\nPASS: Chroma Brain data "
            "was copied to PostgreSQL."
        )


if __name__ == "__main__":
    main()
