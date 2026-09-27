from __future__ import annotations

import csv
from datetime import date
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
REGISTRY_PATH = (
    BASE_DIR
    / "governance"
    / "knowledge_registry.csv"
)


VALID_AUDIENCES = {
    "internal",
    "doctor",
    "customer",
    "shared",
}


VALID_APPROVAL_STATUSES = {
    "draft",
    "approved",
    "expired",
    "rejected",
}


REQUIRED_FIELDS = {
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
}


def _validate_iso_date(
    value,
    field_name,
    registry_id,
):
    value = str(value or "").strip()

    if not value:
        return

    try:
        date.fromisoformat(
            value[:10]
        )
    except ValueError as exc:
        raise ValueError(
            f"{registry_id}: invalid "
            f"{field_name}={value!r}"
        ) from exc


def validate_registry_row(row):
    missing_fields = (
        REQUIRED_FIELDS
        - set(row)
    )

    if missing_fields:
        raise ValueError(
            "Registry row is missing fields: "
            f"{sorted(missing_fields)}"
        )

    registry_id = str(
        row.get(
            "registry_id",
            "",
        )
    ).strip()

    if not registry_id:
        raise ValueError(
            "registry_id cannot be empty"
        )

    source = str(
        row.get(
            "source",
            "",
        )
    ).strip()

    if not source:
        raise ValueError(
            f"{registry_id}: source cannot "
            "be empty"
        )

    content_type = str(
        row.get(
            "content_type",
            "",
        )
    ).strip()

    if not content_type:
        raise ValueError(
            f"{registry_id}: content_type "
            "cannot be empty"
        )

    language = str(
        row.get(
            "language",
            "",
        )
    ).strip()

    if not language:
        raise ValueError(
            f"{registry_id}: language "
            "cannot be empty"
        )

    audience = str(
        row.get(
            "audience",
            "",
        )
    ).strip()

    if audience not in VALID_AUDIENCES:
        raise ValueError(
            f"{registry_id}: invalid "
            f"audience={audience!r}"
        )

    approval_status = str(
        row.get(
            "approval_status",
            "",
        )
    ).strip()

    if (
        approval_status
        not in VALID_APPROVAL_STATUSES
    ):
        raise ValueError(
            f"{registry_id}: invalid "
            "approval_status="
            f"{approval_status!r}"
        )

    _validate_iso_date(
        row.get("review_date"),
        "review_date",
        registry_id,
    )

    _validate_iso_date(
        row.get("expires_at"),
        "expires_at",
        registry_id,
    )

    return True


def load_registry(
    registry_path=REGISTRY_PATH,
):
    registry_path = Path(
        registry_path
    )

    if not registry_path.exists():
        raise FileNotFoundError(
            "Knowledge governance registry "
            f"not found: {registry_path}"
        )

    with registry_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(
            file
        )

        rows = list(reader)

    if not rows:
        raise ValueError(
            "Knowledge governance registry "
            "is empty."
        )

    seen_ids = set()

    for row in rows:
        validate_registry_row(
            row
        )

        registry_id = row[
            "registry_id"
        ]

        if registry_id in seen_ids:
            raise ValueError(
                "Duplicate registry_id: "
                f"{registry_id}"
            )

        seen_ids.add(
            registry_id
        )

    return rows


def get_governance_record(
    source,
    content_type,
    language,
    country="",
    registry_path=REGISTRY_PATH,
):
    """
    Return exactly one approved governance
    decision for a source.

    Matching is based on:
    source + content_type + language + country.
    """
    rows = load_registry(
        registry_path
    )

    source = str(source).strip()
    content_type = str(
        content_type
    ).strip()
    language = str(
        language
    ).strip()
    country = str(
        country or ""
    ).strip()

    matches = [
        row
        for row in rows
        if (
            row["source"].strip()
            == source
            and row[
                "content_type"
            ].strip()
            == content_type
            and row[
                "language"
            ].strip()
            == language
            and row[
                "country"
            ].strip()
            == country
        )
    ]

    if not matches:
        raise KeyError(
            "No governance registry "
            "record found for "
            f"source={source!r}, "
            f"content_type={content_type!r}, "
            f"language={language!r}, "
            f"country={country!r}"
        )

    if len(matches) > 1:
        raise ValueError(
            "Multiple governance records "
            "matched the same source: "
            f"{source!r}"
        )

    return matches[0]


def record_is_indexable(
    record,
    today=None,
):
    """
    Decide whether a registry record may
    enter the searchable Brain index.

    Fail closed:
    - only approved content may be indexed;
    - expired dates are rejected;
    - explicit expired/rejected/draft states
      are rejected.
    """
    validate_registry_row(
        record
    )

    if (
        record["approval_status"]
        != "approved"
    ):
        return False

    expires_at = str(
        record.get(
            "expires_at",
            "",
        )
    ).strip()

    if not expires_at:
        return True

    if today is None:
        today = date.today()

    expiration_date = (
        date.fromisoformat(
            expires_at[:10]
        )
    )

    return (
        expiration_date
        >= today
    )


def governance_metadata(
    record,
):
    """
    Return the governance fields that should
    be attached to indexed passages.
    """
    validate_registry_row(
        record
    )

    return {
        "audience": (
            record["audience"]
        ),
        "approval_status": (
            record[
                "approval_status"
            ]
        ),
        "approver": (
            record["approver"]
        ),
        "knowledge_version": (
            record[
                "knowledge_version"
            ]
        ),
        "review_date": (
            record["review_date"]
        ),
        "expires_at": (
            record["expires_at"]
        ),
        "registry_id": (
            record["registry_id"]
        ),
    }
