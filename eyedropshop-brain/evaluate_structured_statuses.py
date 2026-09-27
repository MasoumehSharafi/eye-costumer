from __future__ import annotations
from uuid import UUID
import csv
from pathlib import Path

from answer_generator import generate_response
from retrieval import retrieve_passages


BASE_DIR = Path(__file__).resolve().parent

CASES_FILE = (
    BASE_DIR
    / "evaluation"
    / "structured_status_eval_v1_1.csv"
)

RESULTS_FILE = (
    BASE_DIR
    / "evaluation"
    / "structured_status_results_v1_1.csv"
)


def load_cases():
    with CASES_FILE.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def add_test_governance_metadata(
    passages,
    status,
    case_id,
):
    """
    Add test-only governance metadata.

    This does not modify ChromaDB or production knowledge.
    """
    if not status:
        return passages

    marker = {
        "id": f"status-test-{case_id}",
        "text": (
            "Synthetic governance marker used only for "
            "structured-status regression testing."
        ),
        "metadata": {
            "response_status": status,
            "source": "structured_status_test",
        },
        "distance": 0.0,
    }

    return [marker] + list(passages)

def is_valid_uuid(value):
    try:
        UUID(str(value))
        return True
    except (ValueError, TypeError, AttributeError):
        return False


def validate_response_contract(response):
    required_keys = {
        "status",
        "answer",
        "evidence",
        "citations",
        "applied_permissions",
        "review",
        "trace_id",
    }

    missing_keys = required_keys - set(response.keys())

    if missing_keys:
        return False, f"missing keys: {sorted(missing_keys)}"

    if not isinstance(response["answer"], str):
        return False, "answer must be a string"

    if not isinstance(response["evidence"], list):
        return False, "evidence must be a list"

    if not isinstance(response["citations"], list):
        return False, "citations must be a list"

    permissions = response["applied_permissions"]

    if not isinstance(permissions, dict):
        return False, "applied_permissions must be a dict"

    required_permission_keys = {
        "audience",
        "language",
        "knowledge_scope",
    }

    missing_permissions = (
        required_permission_keys
        - set(permissions.keys())
    )

    if missing_permissions:
        return (
            False,
            f"missing permission keys: "
            f"{sorted(missing_permissions)}",
        )

    review = response["review"]

    if not isinstance(review, dict):
        return False, "review must be a dict"

    if "required" not in review:
        return False, "review.required missing"

    if "reason" not in review:
        return False, "review.reason missing"

    if not is_valid_uuid(response["trace_id"]):
        return False, "trace_id is not a valid UUID"

    evidence_ids = {
        item.get("source_id")
        for item in response["evidence"]
        if isinstance(item, dict)
    }

    for citation in response["citations"]:
        if not isinstance(citation, dict):
            return False, "citation must be a dict"

        if citation.get("source_id") not in evidence_ids:
            return (
                False,
                "citation source_id is not present in evidence",
            )

        label = citation.get("label")

        if label and f"[{label}]" not in response["answer"]:
            return (
                False,
                f"{label} is not referenced in answer",
            )

    status = response["status"]

    expected_review = status in {
        "conflicting_sources",
        "human_review_required",
    }

    if review["required"] != expected_review:
        return (
            False,
            f"review.required incorrect for status={status}",
        )

    if expected_review and review["reason"] != status:
        return (
            False,
            f"review.reason should be {status}",
        )

    if not expected_review and review["reason"] is not None:
        return (
            False,
            "review.reason should be None",
        )

    return True, ""

def main():
    cases = load_cases()
    results = []

    print()
    print("EyeDropShop structured-status evaluation")
    print("=" * 60)

    for case in cases:
        passages, _ = retrieve_passages(
            question=case["question"],
            language=case["language"],
        )

        forced_status = (
            case.get("forced_metadata_status", "")
            .strip()
        )

        passages = add_test_governance_metadata(
            passages=passages,
            status=forced_status,
            case_id=case["id"],
        )

        response = generate_response(
            question=case["question"],
            language=case["language"],
            passages=passages,
        )

        actual_status = response["status"]
        expected_status = case["expected_status"]

        status_passed = actual_status == expected_status

        contract_passed, contract_error = (
            validate_response_contract(response)
        )

        passed = status_passed and contract_passed

        results.append(
            {
                "id": case["id"],
                "language": case["language"],
                "category": case["category"],
                "question": case["question"],
                "expected_status": expected_status,
                "actual_status": actual_status,
                "passed": passed,
                "contract_passed": contract_passed,
                "contract_error": contract_error,
                "answer": response["answer"],
            }
        )

        symbol = "PASS" if passed else "FAIL"

        print(
            f"{symbol:4} | "
            f"{case['id']} | "
            f"expected={expected_status} | "
            f"actual={actual_status} | "
            f"contract={'PASS' if contract_passed else 'FAIL'}"
        )

        if contract_error:
            print(
                f"       Contract error: {contract_error}"
            )

    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULTS_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "id",
                "language",
                "category",
                "question",
                "expected_status",
                "actual_status",
                "passed",
                "contract_passed",
                "contract_error",
                "answer",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    passed_count = sum(
        result["passed"]
        for result in results
    )

    total = len(results)

    print()
    print("=" * 60)
    print(
        f"Structured-status pass rate: "
        f"{passed_count}/{total} "
        f"({passed_count / total * 100:.1f}%)"
    )

    print(
        f"Results saved to: "
        f"{RESULTS_FILE.relative_to(BASE_DIR)}"
    )

    if passed_count != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
