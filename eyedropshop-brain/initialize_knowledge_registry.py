from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import chromadb


BASE_DIR = Path(__file__).resolve().parent
DATABASE_FOLDER = (
    BASE_DIR
    / "chroma_products_db"
)
COLLECTION_NAME = (
    "eyedropshop_products"
)

REGISTRY_PATH = (
    BASE_DIR
    / "governance"
    / "knowledge_registry.csv"
)


FIELDNAMES = [
    "registry_id",
    "source",
    "content_type",
    "language",
    "country",
    "audience",
    "approval_status",
    "approver",
    "knowledge_version",
    "review_date",
    "expires_at",
    "notes",
]


GOVERNANCE_FIELDS = [
    "audience",
    "approval_status",
    "approver",
    "knowledge_version",
    "review_date",
    "expires_at",
]


def make_registry_id(
    source,
    content_type,
    language,
    country,
):
    raw = "|".join(
        [
            source,
            content_type,
            language,
            country,
        ]
    )

    digest = hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()[:12]

    return (
        f"gov-{digest}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite an existing "
            "knowledge registry."
        ),
    )

    args = parser.parse_args()

    if (
        REGISTRY_PATH.exists()
        and not args.force
    ):
        raise SystemExit(
            "Registry already exists: "
            f"{REGISTRY_PATH}\n"
            "Use --force only if you "
            "intentionally want to rebuild it."
        )

    client = (
        chromadb.PersistentClient(
            path=str(
                DATABASE_FOLDER
            )
        )
    )

    collection = (
        client.get_collection(
            name=COLLECTION_NAME
        )
    )

    result = collection.get(
        include=[
            "metadatas",
        ]
    )

    grouped = {}

    for metadata in (
        result.get(
            "metadatas",
            []
        )
        or []
    ):
        source = str(
            metadata.get(
                "source",
                "",
            )
        ).strip()

        content_type = str(
            metadata.get(
                "content_type",
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
                "",
            )
        ).strip()

        if not source:
            raise ValueError(
                "Found indexed passage "
                "without source metadata."
            )

        if not content_type:
            raise ValueError(
                f"{source}: missing "
                "content_type metadata."
            )

        if not language:
            raise ValueError(
                f"{source}: missing "
                "language metadata."
            )

        key = (
            source,
            content_type,
            language,
            country,
        )

        governance = {
            field: str(
                metadata.get(
                    field,
                    "",
                )
            ).strip()
            for field
            in GOVERNANCE_FIELDS
        }

        existing = grouped.get(
            key
        )

        if existing is None:
            grouped[key] = (
                governance
            )
            continue

        if existing != governance:
            raise ValueError(
                "Conflicting governance "
                "metadata detected for "
                f"{key}.\n"
                f"First: {existing}\n"
                f"Other: {governance}"
            )

    rows = []

    for (
        source,
        content_type,
        language,
        country,
    ), governance in sorted(
        grouped.items()
    ):
        rows.append(
            {
                "registry_id": (
                    make_registry_id(
                        source,
                        content_type,
                        language,
                        country,
                    )
                ),
                "source": source,
                "content_type": (
                    content_type
                ),
                "language": language,
                "country": country,
                "audience": (
                    governance[
                        "audience"
                    ]
                ),
                "approval_status": (
                    governance[
                        "approval_status"
                    ]
                ),
                "approver": (
                    governance[
                        "approver"
                    ]
                ),
                "knowledge_version": (
                    governance[
                        "knowledge_version"
                    ]
                ),
                "review_date": (
                    governance[
                        "review_date"
                    ]
                ),
                "expires_at": (
                    governance[
                        "expires_at"
                    ]
                ),
                "notes": (
                    "Initialized from "
                    "existing Brain metadata"
                ),
            }
        )

    REGISTRY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REGISTRY_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    print(
        "Knowledge registry created:"
    )
    print(
        REGISTRY_PATH
    )
    print(
        f"Registry records: {len(rows)}"
    )


if __name__ == "__main__":
    main()
