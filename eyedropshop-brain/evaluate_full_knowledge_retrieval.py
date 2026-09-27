from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from retrieval_backend import (
    get_retrieval_backend,
    retrieve_passages,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "evaluation/full_knowledge_eval.csv"
DEFAULT_OUTPUT = BASE_DIR / "evaluation/full_knowledge_retrieval_results.csv"


def _clean(value) -> str:
    if value is None:
        return ""
    value = str(value).strip()
    return "" if value.lower() == "nan" else value


def _norm(value) -> str:
    value = _clean(value).lower().replace("\\", "/")
    return " ".join(value.split())


def _source_candidates(metadata: dict) -> list[str]:
    values = []
    for key in (
        "source",
        "source_name",
        "filename",
        "file",
        "title",
        "episode",
    ):
        value = _clean(metadata.get(key))
        if value:
            values.append(value)
    return values


def _source_match(expected_source: str, metadata: dict) -> bool:
    if not expected_source:
        return True

    expected = _norm(expected_source)
    expected_name = _norm(Path(expected_source).name)
    expected_stem = _norm(Path(expected_source).stem)

    for candidate in _source_candidates(metadata):
        c = _norm(candidate)
        c_name = _norm(Path(candidate).name)
        c_stem = _norm(Path(candidate).stem)

        if (
            expected == c
            or expected_name == c_name
            or expected_stem == c_stem
            or expected_stem in c
            or c_stem in expected
        ):
            return True

    return False


def _acceptable_source_hit(
    acceptable_sources: str,
    passages,
) -> bool:
    if not acceptable_sources:
        return False

    allowed_sources = [
        value.strip()
        for value in acceptable_sources.split("|")
        if value.strip()
    ]

    for passage in passages:
        metadata = passage.get("metadata", {})

        for source in allowed_sources:
            if _source_match(source, metadata):
                return True

    return False


def _acceptable_source_and_section_hit(
    acceptable_sources: str,
    expected_section: str,
    passages,
) -> bool:
    if (
        not acceptable_sources
        or not expected_section
    ):
        return False

    allowed_sources = [
        value.strip()
        for value in acceptable_sources.split("|")
        if value.strip()
    ]

    for passage in passages:
        metadata = passage.get("metadata", {})
        passage_section = _clean(
            metadata.get("section")
        )

        if passage_section != expected_section:
            continue

        for source in allowed_sources:
            if _source_match(source, metadata):
                return True

    return False


def evaluate_case(
    row,
    top_k,
    max_distance,
    audience,
):
    question = _clean(row["question"])
    language = _clean(row["language"]) or "en"
    expected_handle = _clean(row.get("expected_handle"))
    expected_section = _clean(row.get("expected_section"))
    expected_source = _clean(row.get("expected_source"))
    acceptable_sources = _clean(
        row.get("acceptable_sources")
    )

    expect_exact = (
        _clean(row.get("expect_exact_match"))
        in {"1", "true", "True"}
    )

    passages, matched_product = retrieve_passages(
        question=question,
        language=language,
        number_of_results=top_k,
        max_distance=max_distance,
        audience=audience,
    )

    detected_handle = (
        matched_product.product_handle
        if matched_product is not None
        else ""
    )

    exact_match_ok = (
        detected_handle == expected_handle
        if expect_exact
        else True
    )

    product_hit = (
        any(
            _clean(
                p.get("metadata", {}).get(
                    "product_handle"
                )
            )
            == expected_handle
            for p in passages
        )
        if expected_handle
        else True
    )

    section_hit = (
        any(
            _clean(
                p.get("metadata", {}).get(
                    "section"
                )
            )
            == expected_section
            for p in passages
        )
        if expected_section
        else True
    )

    exact_source_hit = (
        any(
            _source_match(
                expected_source,
                p.get("metadata", {}),
            )
            for p in passages
        )
        if expected_source
        else True
    )

    # Keep old field name for compatibility.
    source_hit = exact_source_hit

    alternate_source_hit = (
        _acceptable_source_hit(
            acceptable_sources,
            passages,
        )
        if acceptable_sources
        else False
    )

    evidence_hit = (
        (
            exact_source_hit
            or alternate_source_hit
        )
        if expected_source
        else True
    )

    exact_target_hit = False
    alternate_target_hit = False

    if expected_handle and expected_section:
        exact_target_hit = any(
            _clean(
                p.get("metadata", {}).get(
                    "product_handle"
                )
            )
            == expected_handle
            and _clean(
                p.get("metadata", {}).get(
                    "section"
                )
            )
            == expected_section
            for p in passages
        )

        target_hit = exact_target_hit

    elif expected_source and expected_section:
        exact_target_hit = any(
            _source_match(
                expected_source,
                p.get("metadata", {}),
            )
            and _clean(
                p.get("metadata", {}).get(
                    "section"
                )
            )
            == expected_section
            for p in passages
        )

        alternate_target_hit = (
            _acceptable_source_and_section_hit(
                acceptable_sources,
                expected_section,
                passages,
            )
            if acceptable_sources
            else False
        )

        target_hit = (
            exact_target_hit
            or alternate_target_hit
        )

    elif expected_source:
        exact_target_hit = exact_source_hit
        alternate_target_hit = (
            alternate_source_hit
        )
        target_hit = evidence_hit

    elif expected_section:
        exact_target_hit = section_hit
        target_hit = section_hit

    else:
        target_hit = bool(passages)
        exact_target_hit = target_hit

    wrong_product_handles = sorted({
        _clean(
            p.get("metadata", {}).get(
                "product_handle"
            )
        )
        for p in passages
        if (
            _clean(
                p.get("metadata", {}).get(
                    "product_handle"
                )
            )
            and expected_handle
            and _clean(
                p.get("metadata", {}).get(
                    "product_handle"
                )
            )
            != expected_handle
        )
    })

    wrong_product_leak = bool(
        wrong_product_handles
    )

    isolation_ok = (
        not wrong_product_leak
    )

    overall_pass = (
        exact_match_ok
        and target_hit
        and isolation_ok
    )

    retrieved = []

    for p in passages:
        metadata = p.get(
            "metadata",
            {},
        )

        label = (
            _clean(
                metadata.get(
                    "product_handle"
                )
            )
            or _clean(
                metadata.get("source")
            )
            or _clean(
                metadata.get("title")
            )
            or _clean(
                metadata.get("episode")
            )
            or "unknown"
        )

        retrieved.append(
            f"{label}::"
            f"{_clean(metadata.get('section'))}::"
            f"{float(p.get('distance', 0.0)):.4f}"
        )

    return {
        "id": _clean(row.get("id")),
        "knowledge_type": _clean(
            row.get("knowledge_type")
        ),
        "language": language,
        "question": question,
        "source_approval": _clean(
            row.get("source_approval")
        ),
        "expected_handle": expected_handle,
        "expected_section": expected_section,
        "expected_source": expected_source,
        "acceptable_sources": acceptable_sources,
        "detected_handle": detected_handle,
        "exact_match_ok": exact_match_ok,
        "product_hit": product_hit,
        "section_hit": section_hit,
        "source_hit": source_hit,
        "exact_source_hit": exact_source_hit,
        "alternate_source_hit": alternate_source_hit,
        "evidence_hit": evidence_hit,
        "exact_target_hit": exact_target_hit,
        "alternate_target_hit": alternate_target_hit,
        "target_hit": target_hit,
        "wrong_product_leak": wrong_product_leak,
        "wrong_product_handles": " | ".join(
            wrong_product_handles
        ),
        "overall_pass": overall_pass,
        "retrieved_count": len(passages),
        "retrieved": " | ".join(
            retrieved
        ),
    }


def _pct(series) -> str:
    if len(series) == 0:
        return "n/a"
    return f"{series.mean():.1%}"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file",
        type=Path,
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--max-distance",
        type=float,
        default=0.85,
    )

    parser.add_argument(
        "--audience",
        default="internal",
    )

    parser.add_argument(
        "--approved-only",
        action="store_true",
        help=(
            "For file-based sources, only score rows marked approved by "
            "the governance registry. Product rows are always included."
        ),
    )

    args = parser.parse_args()

    os.environ[
        "BRAIN_RETRIEVAL_BACKEND"
    ] = "postgres"

    print(
        "Retrieval backend:",
        get_retrieval_backend(),
    )

    df = pd.read_csv(
        args.file,
        keep_default_na=False,
    )

    if "acceptable_sources" not in df.columns:
        df["acceptable_sources"] = ""

    if args.approved_only:
        df = df[
            (
                df["knowledge_type"]
                == "product"
            )
            |
            (
                df["source_approval"]
                == "approved"
            )
        ].copy()

    results = pd.DataFrame([
        evaluate_case(
            row,
            top_k=args.top_k,
            max_distance=args.max_distance,
            audience=args.audience,
        )
        for _, row in df.iterrows()
    ])

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        args.output,
        index=False,
    )

    print(
        "\nFull knowledge retrieval evaluation"
    )
    print("=" * 78)
    print(f"Cases: {len(results)}")
    print(f"Top-k: {args.top_k}")
    print(
        f"Max distance: {args.max_distance}"
    )
    print(f"Audience: {args.audience}")
    print(
        f"Overall pass: "
        f"{_pct(results['overall_pass'])}"
    )
    print(
        f"Target hit: "
        f"{_pct(results['target_hit'])}"
    )
    print(
        "Wrong-product leakage rate: "
        f"{_pct(results['wrong_product_leak'])}"
    )

    print("\nBy knowledge type:")

    for (
        knowledge_type,
        group,
    ) in results.groupby(
        "knowledge_type"
    ):
        print(
            f"  {knowledge_type:12s} "
            f"cases={len(group):4d} "
            f"pass={_pct(group['overall_pass'])} "
            f"target_hit={_pct(group['target_hit'])}"
        )

    product_rows = results[
        results["knowledge_type"]
        == "product"
    ]

    if not product_rows.empty:
        print("\nProduct-specific:")
        print(
            "  Exact product detection: "
            f"{_pct(product_rows['exact_match_ok'])}"
        )
        print(
            "  Product+section target hit: "
            f"{_pct(product_rows['target_hit'])}"
        )
        print(
            "  Wrong-product leakage: "
            f"{_pct(product_rows['wrong_product_leak'])}"
        )

    transcript_rows = results[
        results["knowledge_type"]
        == "transcript"
    ]

    if not transcript_rows.empty:
        print("\nTranscript-specific:")
        print(
            "  Exact-source recall: "
            f"{_pct(transcript_rows['exact_source_hit'])}"
        )
        print(
            "  Evidence recall: "
            f"{_pct(transcript_rows['evidence_hit'])}"
        )
        print(
            "  Exact target hit: "
            f"{_pct(transcript_rows['exact_target_hit'])}"
        )
        print(
            "  Exact-or-alternate target hit: "
            f"{_pct(transcript_rows['target_hit'])}"
        )

    failed = results[
        ~results["overall_pass"]
    ]

    if not failed.empty:
        print(
            f"\nFAILED CASES "
            f"({len(failed)}):"
        )

        for _, row in (
            failed
            .head(100)
            .iterrows()
        ):
            print(
                f"- [{row['id']}] "
                f"{row['knowledge_type']} | "
                f"{row['question']}"
            )

            if row["expected_handle"]:
                print(
                    "  expected handle/section:",
                    row["expected_handle"],
                    "/",
                    row["expected_section"],
                )
                print(
                    "  detected handle:",
                    row["detected_handle"]
                    or "<none>",
                )

            if row["expected_source"]:
                print(
                    "  expected source:",
                    row["expected_source"],
                )

            if row["acceptable_sources"]:
                print(
                    "  acceptable sources:",
                    row["acceptable_sources"],
                )

            if row["wrong_product_handles"]:
                print(
                    "  WRONG PRODUCT LEAK:",
                    row["wrong_product_handles"],
                )

            print(
                "  exact_source_hit:",
                row["exact_source_hit"],
            )

            print(
                "  alternate_source_hit:",
                row["alternate_source_hit"],
            )

            print(
                "  evidence_hit:",
                row["evidence_hit"],
            )

            print(
                "  retrieved:",
                row["retrieved"]
                or "<none>",
            )

    print(
        f"\nDetailed results saved to: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()
