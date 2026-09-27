from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from retrieval import MAX_DISTANCE, NUMBER_OF_RESULTS, retrieve_passages


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_EVAL_FILE = BASE_DIR / "evaluation/retrieval_eval.csv"
DEFAULT_OUTPUT_FILE = BASE_DIR / "evaluation/retrieval_eval_results.csv"


def evaluate_case(row, top_k, max_distance):
    passages, matched_product = retrieve_passages(
        question=row["question"],
        language=row["language"],
        number_of_results=top_k,
        max_distance=max_distance,
    )

    expected_handle = str(row.get("expected_handle", "")).strip()
    expected_section = str(row.get("expected_section", "")).strip()
    expect_exact_match = bool(int(row.get("expect_exact_match", 0)))

    retrieved_handles = [
        str(p["metadata"].get("product_handle", ""))
        for p in passages
    ]
    retrieved_sections = [
        str(p["metadata"].get("section", ""))
        for p in passages
    ]

    exact_handle = matched_product.product_handle if matched_product else ""

    exact_match_ok = (
        exact_handle == expected_handle
        if expect_exact_match
        else matched_product is None
    )

    product_hit = (
        expected_handle in retrieved_handles
        if expected_handle
        else True
    )

    section_hit = (
        expected_section in retrieved_sections
        if expected_section
        else True
    )

    if expected_handle and expected_section:
        target_passage_hit = any(
            p["metadata"].get("product_handle") == expected_handle
            and p["metadata"].get("section") == expected_section
            for p in passages
        )

        # Mixed product + policy questions are successful when both pieces
        # are present, even though they necessarily come from two passages.
        if expected_section in {"shipping", "returns"}:
            target_passage_hit = product_hit and section_hit
    else:
        target_passage_hit = section_hit

    expected_rank = ""
    for index, passage in enumerate(passages, start=1):
        metadata = passage["metadata"]

        if expected_section in {"shipping", "returns"}:
            # For mixed product + policy questions, rank the policy passage.
            handle_ok = True
        else:
            handle_ok = (
                not expected_handle
                or metadata.get("product_handle") == expected_handle
            )

        section_ok = (
            not expected_section
            or metadata.get("section") == expected_section
        )
        if handle_ok and section_ok:
            expected_rank = index
            break

    return {
        "id": row["id"],
        "language": row["language"],
        "question": row["question"],
        "expected_handle": expected_handle,
        "expected_section": expected_section,
        "detected_handle": exact_handle,
        "exact_match_ok": exact_match_ok,
        "product_hit_at_k": product_hit,
        "section_hit_at_k": section_hit,
        "target_hit_at_k": target_passage_hit,
        "expected_rank": expected_rank,
        "retrieved": " | ".join(
            f"{p['metadata'].get('product_handle') or p['metadata'].get('title')}"
            f"::{p['metadata'].get('section')}"
            f"::{p['distance']:.4f}"
            for p in passages
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--top-k", type=int, default=NUMBER_OF_RESULTS)
    parser.add_argument("--max-distance", type=float, default=MAX_DISTANCE)
    args = parser.parse_args()

    eval_df = pd.read_csv(args.file, keep_default_na=False)

    rows = [
        evaluate_case(row, args.top_k, args.max_distance)
        for _, row in eval_df.iterrows()
    ]
    results = pd.DataFrame(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)

    print("\nRetrieval evaluation")
    print("=" * 60)
    print(f"Cases: {len(results)}")
    print(f"Top-k: {args.top_k}")
    print(f"Max distance: {args.max_distance}")
    print(
        "Exact product detection accuracy: "
        f"{results['exact_match_ok'].mean():.1%}"
    )
    print(
        "Expected product hit@k: "
        f"{results['product_hit_at_k'].mean():.1%}"
    )
    print(
        "Expected section hit@k: "
        f"{results['section_hit_at_k'].mean():.1%}"
    )
    print(
        "Target retrieval hit@k: "
        f"{results['target_hit_at_k'].mean():.1%}"
    )

    failed = results[~results["target_hit_at_k"]]
    if not failed.empty:
        print("\nFailed target cases:")
        for _, row in failed.iterrows():
            print(f"- [{row['id']}] {row['question']}")
            print(f"  Retrieved: {row['retrieved']}")

    print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()