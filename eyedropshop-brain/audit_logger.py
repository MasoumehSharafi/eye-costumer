from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
AUDIT_LOG_PATH = BASE_DIR / "logs" / "audit_log.jsonl"


def log_audit_event(
    question,
    language,
    response,
):
    """
    Persist one structured Brain decision as a JSONL audit record.

    Full retrieved passage text is intentionally not stored.
    We keep only source identifiers and response-level information
    needed to reconstruct the decision.
    """
    AUDIT_LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    record = {
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "trace_id": response.get("trace_id"),
        "question": question,
        "language": language,
        "status": response.get("status"),
        "answer": response.get("answer"),
        "evidence_ids": evidence_ids,
        "citation_ids": citation_ids,
        "applied_permissions": response.get(
            "applied_permissions",
            {},
        ),
        "review": response.get(
            "review",
            {},
        ),
    }

    with AUDIT_LOG_PATH.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )

    return record
