from __future__ import annotations

import csv
import os
from pathlib import Path
from uuid import UUID

from fastapi.testclient import TestClient

from answer_generator import generate_response
import psycopg

from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from postgres_retrieval import (
    POSTGRES_DSN,
    load_embedding_model,
)
from retrieval_backend import (
    get_retrieval_backend,
    retrieve_passages,
)


BASE_DIR = Path(__file__).resolve().parent
RESULTS_PATH = (
    BASE_DIR
    / "evaluation"
    / "brain_system_results_v1_1.csv"
)


results = []


def record(
    test_id,
    category,
    passed,
    details,
):
    results.append(
        {
            "test_id": test_id,
            "category": category,
            "passed": passed,
            "details": details,
        }
    )

    print(
        f"{'PASS' if passed else 'FAIL'}"
        f" | {test_id}"
        f" | {category}"
        f" | {details}"
    )


def valid_uuid(value):
    try:
        UUID(str(value))
        return True
    except Exception:
        return False


def validate_contract(
    response,
    expected_status,
    expected_audience,
):
    required_keys = {
        "status",
        "answer",
        "evidence",
        "citations",
        "applied_permissions",
        "review",
        "trace_id",
    }

    if not required_keys <= set(response):
        return False, "missing response fields"

    if response["status"] != expected_status:
        return (
            False,
            (
                f"expected status={expected_status}, "
                f"got={response['status']}"
            ),
        )

    if not isinstance(
        response["answer"],
        str,
    ):
        return False, "answer is not a string"

    if not isinstance(
        response["evidence"],
        list,
    ):
        return False, "evidence is not a list"

    if not isinstance(
        response["citations"],
        list,
    ):
        return False, "citations is not a list"

    permissions = response[
        "applied_permissions"
    ]

    if (
        permissions.get("audience")
        != expected_audience
    ):
        return (
            False,
            "incorrect applied audience",
        )

    if not valid_uuid(
        response["trace_id"]
    ):
        return False, "invalid trace_id"

    evidence_ids = {
        item.get("source_id")
        for item in response["evidence"]
        if isinstance(item, dict)
    }

    for citation in response[
        "citations"
    ]:
        if (
            citation.get("source_id")
            not in evidence_ids
        ):
            return (
                False,
                "citation not present in evidence",
            )

        label = citation.get("label")

        if (
            label
            and f"[{label}]"
            not in response["answer"]
        ):
            return (
                False,
                "citation label missing from answer",
            )

    expected_review = (
        expected_status
        in {
            "conflicting_sources",
            "human_review_required",
        }
    )

    if (
        response["review"].get(
            "required"
        )
        != expected_review
    ):
        return (
            False,
            "incorrect review requirement",
        )

    return True, "contract valid"


def run_real_answer_tests():
    print(
        "\n=== Brain answer tests ==="
    )

    cases = [
        {
            "id": "brain_001",
            "question": (
                "Can I return unopened "
                "LUMIFY within 30 days?"
            ),
            "language": "en",
            "audience": "internal",
            "expected_status": "answerable",
            "require_evidence": True,
            "require_citations": True,
        },
        {
            "id": "brain_002",
            "question": (
                "Do you ship to the "
                "United States?"
            ),
            "language": "en",
            "audience": "internal",
            "expected_status": (
                "insufficient_knowledge"
            ),
            "require_evidence": False,
            "require_citations": False,
            "required_answer_terms": [
                "Customer Happiness",
                "info@eyedropshop.com",
            ],
        },
        {
            "id": "brain_003",
            "question": (
                "My eyes hurt. "
                "What eye drops should I use?"
            ),
            "language": "en",
            "audience": "internal",
            "expected_status": "restricted",
            "require_evidence": False,
            "require_citations": False,
            "required_answer_terms": [
                "eye-care professional",
            ],
        },

        {
            "id": "brain_016",
            "question": (
                "Why does heat therapy help "
                "with dry eye?"
            ),
            "language": "en",
            "audience": "doctor",
            "expected_status": "answerable",
            "require_evidence": True,
            "require_citations": True,
            "required_content_type": "transcript",
        },
        {
            "id": "brain_017",
            "question": (
                "Why does heat therapy help "
                "with dry eye?"
            ),
            "language": "en",
            "audience": "doctor",
            "expected_status": "answerable",
            "require_evidence": True,
            "require_citations": False,
            "required_source_prefix": "transcript-",
        },
        {
            "id": "brain_018",
            "question": (
                "Why does heat therapy help "
                "with dry eye?"
            ),
            "language": "en",
            "audience": "doctor",
            "expected_status": "answerable",
            "require_evidence": True,
            "require_citations": True,
            "required_citation_prefix": "transcript-",
        },
        {
            "id": "brain_019",
            "question": (
                "Why does heat therapy help "
                "with dry eye?"
            ),
            "language": "en",
            "audience": "doctor",
            "expected_status": "answerable",
            "require_evidence": True,
            "require_citations": False,
            "required_evidence_audience": "shared",
        },

        {
            "id": "brain_020",
            "question": "What are meibomian glands?",
            "language": "en",
            "audience": "customer",
            "expected_status": "answerable",
            "require_evidence": True,
            "require_citations": True,
            "required_content_type": "transcript",
            "required_evidence_audience": "shared",
        },

        {
            "id": "brain_021",
            "question": (
                "Which product is best "
                "for my dry eye?"
            ),
            "language": "en",
            "audience": "customer",
            "expected_status": "restricted",
            "require_evidence": False,
            "require_citations": False,
            "required_answer_terms": [
                "eye-care professional",
            ],
        },
    ]

    for case in cases:
        passages, _ = retrieve_passages(
            question=case["question"],
            language=case["language"],
            audience=case["audience"],
        )

        response = generate_response(
            question=case["question"],
            language=case["language"],
            passages=passages,
            audience=case["audience"],
        )

        passed, reason = (
            validate_contract(
                response=response,
                expected_status=case[
                    "expected_status"
                ],
                expected_audience=case[
                    "audience"
                ],
            )
        )

        if (
            passed
            and case[
                "require_evidence"
            ]
            and not response["evidence"]
        ):
            passed = False
            reason = "expected evidence"

        if (
            passed
            and case[
                "require_citations"
            ]
            and not response["citations"]
        ):
            passed = False
            reason = "expected citations"

        required_terms = case.get(
            "required_answer_terms",
            [],
        )

        for term in required_terms:
            if term.lower() not in response[
                "answer"
            ].lower():
                passed = False
                reason = (
                    "missing required answer term: "
                    f"{term}"
                )
                break
        required_content_type = case.get(
            "required_content_type"
        )

        if passed and required_content_type:
            if not any(
                item.get(
                    "metadata",
                    {},
                ).get("content_type")
                == required_content_type
                for item in response["evidence"]
            ):
                passed = False
                reason = (
                    "expected evidence content_type="
                    f"{required_content_type}"
                )


        required_source_prefix = case.get(
            "required_source_prefix"
        )

        if passed and required_source_prefix:
            if not any(
                str(
                    item.get(
                        "source_id",
                        "",
                    )
                ).startswith(
                    required_source_prefix
                )
                for item in response["evidence"]
            ):
                passed = False
                reason = (
                    "expected evidence source prefix="
                    f"{required_source_prefix}"
                )


        required_citation_prefix = case.get(
            "required_citation_prefix"
        )

        if passed and required_citation_prefix:
            if not any(
                str(
                    item.get(
                        "source_id",
                        "",
                    )
                ).startswith(
                    required_citation_prefix
                )
                for item in response["citations"]
            ):
                passed = False
                reason = (
                    "expected citation source prefix="
                    f"{required_citation_prefix}"
                )


        required_evidence_audience = case.get(
            "required_evidence_audience"
        )

        if passed and required_evidence_audience:
            if not any(
                item.get(
                    "metadata",
                    {},
                ).get("audience")
                == required_evidence_audience
                for item in response["evidence"]
            ):
                passed = False
                reason = (
                    "expected evidence audience="
                    f"{required_evidence_audience}"
                )
        record(
            case["id"],
            "brain_response",
            passed,
            reason,
        )


def run_review_status_tests():
    print(
        "\n=== Review-status tests ==="
    )

    cases = [
        (
            "brain_004",
            "conflicting_sources",
        ),
        (
            "brain_005",
            "human_review_required",
        ),
    ]

    for test_id, status in cases:
        passages = [
            {
                "id": (
                    f"system-test-{status}"
                ),
                "text": (
                    "Synthetic governed "
                    "evaluation evidence."
                ),
                "distance": 0.0,
                "metadata": {
                    "title": (
                        "System Evaluation"
                    ),
                    "section": "test",
                    "response_status": status,
                    "approval_status": (
                        "approved"
                    ),
                    "audience": (
                        "internal"
                    ),
                },
            }
        ]

        response = generate_response(
            question=(
                "Synthetic governance test"
            ),
            language="en",
            passages=passages,
            audience="internal",
        )

        passed, reason = (
            validate_contract(
                response=response,
                expected_status=status,
                expected_audience=(
                    "internal"
                ),
            )
        )

        record(
            test_id,
            "review_status",
            passed,
            reason,
        )

def run_permission_tests():
    print(
        "\n=== Knowledge permission tests ==="
    )

    embedding_model = load_embedding_model()

    test_records = [
        {
            "registry_id": "system-source-doctor",
            "passage_id": "system-perm-doctor",
            "text": (
                "SYSTEM_PERMISSION_MARKER "
                "doctor only"
            ),
            "audience": "doctor",
            "approval": "approved",
            "expires_at": None,
        },
        {
            "registry_id": "system-source-customer",
            "passage_id": "system-perm-customer",
            "text": (
                "SYSTEM_PERMISSION_MARKER "
                "customer only"
            ),
            "audience": "customer",
            "approval": "approved",
            "expires_at": None,
        },
        {
            "registry_id": "system-source-internal",
            "passage_id": "system-perm-internal",
            "text": (
                "SYSTEM_PERMISSION_MARKER "
                "internal only"
            ),
            "audience": "internal",
            "approval": "approved",
            "expires_at": None,
        },
        {
            "registry_id": "system-source-shared",
            "passage_id": "system-perm-shared",
            "text": (
                "SYSTEM_PERMISSION_MARKER "
                "shared"
            ),
            "audience": "shared",
            "approval": "approved",
            "expires_at": None,
        },
        {
            "registry_id": "system-source-draft",
            "passage_id": "system-perm-draft",
            "text": (
                "SYSTEM_DRAFT_MARKER "
                "must never be retrieved"
            ),
            "audience": "internal",
            "approval": "draft",
            "expires_at": None,
        },
        {
            "registry_id": "system-source-expired",
            "passage_id": "system-perm-expired",
            "text": (
                "SYSTEM_EXPIRED_MARKER "
                "must never be retrieved"
            ),
            "audience": "internal",
            "approval": "approved",
            "expires_at": "2020-01-01",
        },
    ]

    documents = [
        item["text"]
        for item in test_records
    ]

    embeddings = embedding_model.encode(
        documents,
        normalize_embeddings=True,
    ).tolist()

    passage_ids = {
        item["passage_id"]
        for item in test_records
    }

    registry_ids = [
        item["registry_id"]
        for item in test_records
    ]

    with psycopg.connect(
        POSTGRES_DSN,
        autocommit=True,
    ) as conn:
        register_vector(conn)

        try:
            for item, embedding in zip(
                test_records,
                embeddings,
            ):
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
                        knowledge_version,
                        expires_at,
                        notes
                    )
                    VALUES (
                        %s,
                        %s,
                        'system_evaluation',
                        'en',
                        'CA',
                        %s,
                        %s,
                        'system-evaluation',
                        %s,
                        'Temporary system evaluation'
                    )
                    """,
                    (
                        item["registry_id"],
                        item["registry_id"],
                        item["audience"],
                        item["approval"],
                        item["expires_at"],
                    ),
                )

                metadata = {
                    "language": "en",
                    "section": "description",
                    "content_type": (
                        "system_evaluation"
                    ),
                    "approval_status": (
                        item["approval"]
                    ),
                    "audience": (
                        item["audience"]
                    ),
                    "knowledge_version": (
                        "system-evaluation"
                    ),
                    "registry_id": (
                        item["registry_id"]
                    ),
                    "title": (
                        "System Evaluation"
                    ),
                    "source": (
                        item["registry_id"]
                    ),
                    "expires_at": (
                        item["expires_at"]
                        or ""
                    ),
                }

                conn.execute(
                    """
                    INSERT INTO knowledge_passages (
                        passage_id,
                        source_registry_id,
                        section,
                        content,
                        metadata,
                        embedding
                    )
                    VALUES (
                        %s,
                        %s,
                        'description',
                        %s,
                        %s,
                        %s
                    )
                    """,
                    (
                        item["passage_id"],
                        item["registry_id"],
                        item["text"],
                        Jsonb(metadata),
                        Vector(
                            [
                                float(value)
                                for value
                                in embedding
                            ]
                        ),
                    ),
                )

            permission_cases = {
                "doctor": {
                    "allowed": {
                        "system-perm-doctor",
                        "system-perm-shared",
                    },
                    "blocked": {
                        "system-perm-customer",
                        "system-perm-internal",
                    },
                },
                "customer": {
                    "allowed": {
                        "system-perm-customer",
                        "system-perm-shared",
                    },
                    "blocked": {
                        "system-perm-doctor",
                        "system-perm-internal",
                    },
                },
                "internal": {
                    "allowed": {
                        "system-perm-internal",
                        "system-perm-shared",
                    },
                    "blocked": {
                        "system-perm-doctor",
                        "system-perm-customer",
                    },
                },
            }

            test_number = 6

            for (
                audience,
                expectation,
            ) in permission_cases.items():

                passages, _ = retrieve_passages(
                    question=(
                        "SYSTEM_PERMISSION_MARKER"
                    ),
                    language="en",
                    audience=audience,
                    number_of_results=20,
                    max_distance=4.1,
                )

                returned_ids = {
                    passage["id"]
                    for passage in passages
                }

                allowed_present = (
                    expectation["allowed"]
                    <= returned_ids
                )

                blocked_absent = (
                    expectation["blocked"]
                    .isdisjoint(returned_ids)
                )

                passed = (
                    allowed_present
                    and blocked_absent
                )

                record(
                    f"brain_{test_number:03d}",
                    "audience_permission",
                    passed,
                    (
                        f"{audience}: "
                        f"{sorted(returned_ids & passage_ids)}"
                    ),
                )

                test_number += 1

            draft_passages, _ = (
                retrieve_passages(
                    question=(
                        "SYSTEM_DRAFT_MARKER"
                    ),
                    language="en",
                    audience="internal",
                    number_of_results=20,
                    max_distance=4.1,
                )
            )

            draft_ids = {
                item["id"]
                for item in draft_passages
            }

            record(
                "brain_009",
                "approval_governance",
                (
                    "system-perm-draft"
                    not in draft_ids
                ),
                "draft record blocked",
            )

            expired_passages, _ = (
                retrieve_passages(
                    question=(
                        "SYSTEM_EXPIRED_MARKER"
                    ),
                    language="en",
                    audience="internal",
                    number_of_results=20,
                    max_distance=4.1,
                )
            )

            expired_ids = {
                item["id"]
                for item in expired_passages
            }

            record(
                "brain_010",
                "expiration_governance",
                (
                    "system-perm-expired"
                    not in expired_ids
                ),
                "expired record blocked",
            )

        finally:
            conn.execute(
                """
                DELETE FROM knowledge_passages
                WHERE passage_id = ANY(%s::text[])
                """,
                (
                    list(passage_ids),
                ),
            )

            conn.execute(
                """
                DELETE FROM knowledge_sources
                WHERE registry_id = ANY(%s::text[])
                """,
                (
                    registry_ids,
                ),
            )

            print(
                "Temporary PostgreSQL "
                "governance records removed."
            )

def run_api_security_tests():
    print(
        "\n=== API security tests ==="
    )

    # Temporary test credentials exist
    # only in this Python process.
    os.environ[
        "BRAIN_INTERNAL_API_KEY"
    ] = "system-test-internal-key"

    os.environ[
        "BRAIN_DOCTOR_API_KEY"
    ] = "system-test-doctor-key"

    os.environ[
        "BRAIN_CUSTOMER_API_KEY"
    ] = "system-test-customer-key"

    from brain_api import app

    client = TestClient(app)

    payload = {
        "question": (
            "Can I return unopened "
            "LUMIFY within 30 days?"
        ),
        "language": "en",
    }

    response = client.post(
        "/v1/answer",
        json=payload,
    )

    record(
        "brain_011",
        "api_authentication",
        response.status_code == 401,
        (
            "anonymous status="
            f"{response.status_code}"
        ),
    )

    response = client.post(
        "/v1/answer",
        headers={
            "X-Brain-Role": "internal",
            "X-Brain-Key": "wrong-key",
        },
        json=payload,
    )

    record(
        "brain_012",
        "api_authentication",
        response.status_code == 401,
        (
            "invalid-key status="
            f"{response.status_code}"
        ),
    )

    response = client.post(
        "/v1/answer",
        headers={
            "X-Brain-Role": "internal",
            "X-Brain-Key": (
                "system-test-internal-key"
            ),
        },
        json=payload,
    )

    valid_internal = (
        response.status_code == 200
        and response.json()
        .get(
            "applied_permissions",
            {},
        )
        .get("audience")
        == "internal"
    )

    record(
        "brain_013",
        "api_authorization",
        valid_internal,
        (
            "valid internal "
            f"status={response.status_code}"
        ),
    )

    spoof_payload = dict(payload)
    spoof_payload[
        "audience"
    ] = "doctor"

    response = client.post(
        "/v1/answer",
        headers={
            "X-Brain-Role": "internal",
            "X-Brain-Key": (
                "system-test-internal-key"
            ),
        },
        json=spoof_payload,
    )

    record(
        "brain_014",
        "api_authorization",
        response.status_code == 422,
        (
            "audience spoof "
            f"status={response.status_code}"
        ),
    )

    response = client.post(
        "/v1/answer",
        headers={
            "X-Brain-Role": "doctor",
            "X-Brain-Key": (
                "system-test-doctor-key"
            ),
        },
        json=payload,
    )

    doctor_body = (
        response.json()
        if response.status_code == 200
        else {}
    )

    evidence_ids = {
        item.get("source_id")
        for item in doctor_body.get(
            "evidence",
            []
        )
    }

    valid_doctor = (
        response.status_code == 200
        and doctor_body.get(
            "applied_permissions",
            {},
        ).get("audience")
        == "doctor"
        and "system-perm-internal"
            not in evidence_ids
    )

    record(
        "brain_015",
        "api_authorization",
        valid_doctor,
        (
            "doctor cannot access "
            "internal-only knowledge"
        ),
    )


def save_results():
    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULTS_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "test_id",
                "category",
                "passed",
                "details",
            ],
        )

        writer.writeheader()
        writer.writerows(results)


def main():

    backend = get_retrieval_backend()

    print(
        "Active retrieval backend:",
        backend,
    )

    if backend != "postgres":
        raise RuntimeError(
            "Full PostgreSQL system evaluation "
            "requires "
            "BRAIN_RETRIEVAL_BACKEND=postgres."
        )
    run_real_answer_tests()
    run_review_status_tests()
    run_permission_tests()
    run_api_security_tests()

    save_results()

    passed = sum(
        bool(item["passed"])
        for item in results
    )

    total = len(results)

    print(
        "\n=============================="
    )

    print(
        f"Brain system pass rate: "
        f"{passed}/{total} "
        f"({100 * passed / total:.1f}%)"
    )

    print(
        f"Results saved to: "
        f"{RESULTS_PATH}"
    )

    print(
        "=============================="
    )


if __name__ == "__main__":
    main()
