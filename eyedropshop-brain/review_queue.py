from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


BASE_DIR = Path(__file__).resolve().parent
REVIEW_QUEUE_PATH = BASE_DIR / "logs" / "review_queue.csv"

REVIEW_STATUSES = {
    "open",
    "in_review",
    "resolved",
    "dismissed",
}

FIELDNAMES = [
    "review_id",
    "created_at_utc",
    "updated_at_utc",
    "resolved_at_utc",
    "trace_id",
    "question",
    "language",
    "status",
    "reason",
    "evidence_ids",
    "citation_ids",
    "review_status",
    "reviewer",
    "reviewer_notes",
    "resolution",
]


def _utc_now():
    return datetime.now(
        timezone.utc
    ).isoformat()


def _read_rows():
    if not REVIEW_QUEUE_PATH.exists():
        return []

    with REVIEW_QUEUE_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def _write_rows(rows):
    REVIEW_QUEUE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = REVIEW_QUEUE_PATH.with_suffix(
        ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES,
        )
        writer.writeheader()
        writer.writerows(rows)

    temporary_path.replace(
        REVIEW_QUEUE_PATH
    )


def enqueue_review_item(
    question,
    language,
    response,
):
    """
    Add a Brain decision to the human-review queue.

    Only responses whose structured review field says
    review.required=True are added.
    """
    review = response.get("review", {})

    if not review.get("required"):
        return None

    trace_id = str(
        response.get("trace_id", "")
    ).strip()

    if not trace_id:
        raise ValueError(
            "A review item requires a trace_id."
        )

    rows = _read_rows()

    # Prevent duplicate queue entries for the same Brain response.
    for row in rows:
        if row.get("trace_id") == trace_id:
            return row

    evidence_ids = [
        item.get("source_id")
        for item in response.get("evidence", [])
        if isinstance(item, dict)
        and item.get("source_id")
    ]

    citation_ids = [
        item.get("source_id")
        for item in response.get("citations", [])
        if isinstance(item, dict)
        and item.get("source_id")
    ]

    now = _utc_now()

    record = {
        "review_id": str(uuid4()),
        "created_at_utc": now,
        "updated_at_utc": now,
        "resolved_at_utc": "",
        "trace_id": trace_id,
        "question": question,
        "language": language,
        "status": response.get("status", ""),
        "reason": review.get("reason", ""),
        "evidence_ids": json.dumps(
            evidence_ids,
            ensure_ascii=False,
        ),
        "citation_ids": json.dumps(
            citation_ids,
            ensure_ascii=False,
        ),
        "review_status": "open",
        "reviewer": "",
        "reviewer_notes": "",
        "resolution": "",
    }

    rows.append(record)
    _write_rows(rows)

    return record


def load_review_items(
    review_status=None,
):
    rows = _read_rows()

    if review_status is None:
        return rows

    if review_status not in REVIEW_STATUSES:
        raise ValueError(
            f"Invalid review status: {review_status}"
        )

    return [
        row
        for row in rows
        if row.get("review_status")
        == review_status
    ]


def update_review_item(
    review_id,
    review_status,
    reviewer="",
    reviewer_notes="",
    resolution="",
):
    """
    Update one human-review item.
    """
    if review_status not in REVIEW_STATUSES:
        raise ValueError(
            f"Invalid review status: {review_status}"
        )

    rows = _read_rows()

    updated = None
    now = _utc_now()

    for row in rows:
        if row.get("review_id") != review_id:
            continue

        row["review_status"] = review_status
        row["updated_at_utc"] = now
        row["reviewer"] = reviewer
        row["reviewer_notes"] = reviewer_notes
        row["resolution"] = resolution

        if review_status in {
            "resolved",
            "dismissed",
        }:
            row["resolved_at_utc"] = now
        else:
            row["resolved_at_utc"] = ""

        updated = dict(row)
        break

    if updated is None:
        raise ValueError(
            f"Unknown review_id: {review_id}"
        )

    _write_rows(rows)

    return updated
