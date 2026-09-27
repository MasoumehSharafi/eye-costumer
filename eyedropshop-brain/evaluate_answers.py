from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import ollama
import pandas as pd

from answer_generator import (
    EN_FALLBACK,
    FR_FALLBACK,
    OLLAMA_MODEL,
    generate_answer,
)
from product_matcher import detect_ambiguous_product
from retrieval import retrieve_passages


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_EVAL_FILE = BASE_DIR / "evaluation/retrieval_eval.csv"
DEFAULT_CSV_OUTPUT = BASE_DIR / "evaluation/answer_eval_results.csv"
DEFAULT_JSON_OUTPUT = BASE_DIR / "evaluation/answer_eval_details.json"

SOURCE_RE = re.compile(r"\[Source\s+(\d+)\]", re.IGNORECASE)


def _build_evidence(passages):
    parts = []

    for index, passage in enumerate(passages, start=1):
        metadata = passage["metadata"]

        source_name = (
            metadata.get("product_title")
            or metadata.get("title")
            or metadata.get("product_handle")
            or metadata.get("source")
            or "Unknown source"
        )

        parts.append(
            f"[Source {index}]\n"
            f"Title: {source_name}\n"
            f"Section: {metadata.get('section', '')}\n"
            f"Content: {passage['text']}"
        )

    return "\n\n".join(parts)


def _extract_json(text):
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])

        raise


def _citation_checks(answer, number_of_sources):
    cited = [int(value) for value in SOURCE_RE.findall(answer)]

    invalid = sorted(
        {
            source_number
            for source_number in cited
            if source_number < 1 or source_number > number_of_sources
        }
    )

    return {
        "cited_sources": sorted(set(cited)),
        "invalid_citations": invalid,
        "citation_numbers_valid": len(invalid) == 0,
    }


def _judge_prompt(question, language, evidence, answer):
    if language == "fr":
        return f"""
Évaluez strictement la fidélité factuelle de cette réponse EyeDropShop.

QUESTION
{question}

INFORMATIONS APPROUVÉES
{evidence}

RÉPONSE
{answer}

Évaluez les affirmations EXACTEMENT telles qu'elles sont écrites.

Règles très importantes :
1. Préservez toujours la négation et la modalité.
   - « les sources ne disent pas X » n'est PAS l'affirmation « X ».
   - « je ne dispose pas de suffisamment d'informations » n'est PAS une affirmation factuelle sur le produit.
2. N'inventez jamais une affirmation positive à partir d'une phrase négative.
3. N'utilisez aucune connaissance externe ni inférence par omission.
4. Une paraphrase ou traduction fidèle est acceptable.
5. Verdicts autorisés : supported, partial, unsupported, allowed_safety, control, prompt_leak.
6. allowed_safety et control ne nécessitent pas de citation. Pour ces verdicts, mettez citation_match=true et supporting_sources=[].
7. prompt_leak = texte qui expose une instruction interne, une règle système, une consigne au modèle ou un processus d'évaluation.
8. Pour supported/partial/unsupported, vérifiez si les citations [Source N] réellement attachées à l'affirmation la soutiennent.
9. Ignorez les introductions purement stylistiques.

Retournez UNIQUEMENT du JSON valide :
{{
  "claims": [
    {{
      "claim": "...",
      "verdict": "supported|partial|unsupported|allowed_safety|control|prompt_leak",
      "supporting_sources": [1],
      "citation_match": true,
      "reason": "..."
    }}
  ],
  "summary": "..."
}}
"""
    return f"""
Strictly evaluate factual grounding for this EyeDropShop answer.

QUESTION
{question}

APPROVED INFORMATION
{evidence}

ANSWER
{answer}

Evaluate claims EXACTLY as they are written.

Very important rules:
1. Always preserve negation and modality.
   - "the sources do not say X" is NOT the claim "X".
   - "I don't have enough approved information" is NOT a factual claim about the product.
2. Never turn a negative statement into a positive claim.
3. Use no external knowledge and do not infer facts from omission.
4. Faithful paraphrases/translations are acceptable.
5. Allowed verdicts: supported, partial, unsupported, allowed_safety, control, prompt_leak.
6. allowed_safety and control do NOT require citations. For those verdicts, set citation_match=true and supporting_sources=[].
7. prompt_leak = text that exposes an internal instruction, system rule, model directive, or evaluation process.
8. For supported/partial/unsupported, verify whether the [Source N] citations actually attached to the claim support it.
9. Ignore purely stylistic introductory text.

Return ONLY valid JSON:
{{
  "claims": [
    {{
      "claim": "...",
      "verdict": "supported|partial|unsupported|allowed_safety|control|prompt_leak",
      "supporting_sources": [1],
      "citation_match": true,
      "reason": "..."
    }}
  ],
  "summary": "..."
}}
"""


def judge_grounding(question, language, passages, answer):
    evidence = _build_evidence(passages)
    prompt = _judge_prompt(question, language, evidence, answer)

    last_error = None
    last_raw = ""

    for attempt in range(2):
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return only compact valid JSON. "
                        "Do not include markdown or commentary."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            think=False,
            keep_alive="30m",
            options={
                "temperature": 0.0,
                "num_ctx": 8192,
                "num_predict": 1200,
            },
        )

        last_raw = response["message"]["content"]

        try:
            parsed = _extract_json(last_raw)

            if "claims" not in parsed or not isinstance(parsed["claims"], list):
                raise ValueError("Judge JSON does not contain a claims list.")

            return parsed, last_raw
        except Exception as exc:
            last_error = exc
            prompt = (
                _judge_prompt(question, language, evidence, answer)
                + "\nIMPORTANT: Your previous response was invalid JSON. "
                  "Return a smaller JSON object with no markdown."
            )

    raise ValueError(
        f"Judge returned invalid JSON after retry: {last_error}"
    )


def evaluate_case(row):
    question = str(row["question"])
    language = str(row["language"])

    passages, matched_product = retrieve_passages(
        question=question,
        language=language,
    )

    answer = generate_answer(
        question=question,
        language=language,
        passages=passages,
    )

    citation_checks = _citation_checks(
        answer=answer,
        number_of_sources=len(passages),
    )

    ambiguity = detect_ambiguous_product(
        question,
        language,
    )

    control_response = (
        answer.strip() in {EN_FALLBACK, FR_FALLBACK}
        or ambiguity is not None
    )

    # "I don't have enough approved information" and clarification
    # questions are control/safety responses, not factual claims that need
    # source citations. Their appropriateness is evaluated separately by the
    # robustness behavior evaluator.
    if control_response:
        grounding = {
            "claims": [],
            "summary": (
                "Control response: safe insufficient-information fallback "
                "or ambiguous-product clarification."
            ),
        }
        raw_judge = ""
        judge_error = ""
    elif passages:
        try:
            grounding, raw_judge = judge_grounding(
                question=question,
                language=language,
                passages=passages,
                answer=answer,
            )
            judge_error = ""
        except Exception as exc:
            grounding = {
                "claims": [],
                "summary": "",
            }
            raw_judge = ""
            judge_error = str(exc)
    else:
        grounding = {
            "claims": [],
            "summary": (
                "No evidence was retrieved; answer used a non-factual "
                "response."
            ),
        }
        raw_judge = ""
        judge_error = ""

    claims = grounding.get("claims", [])

    verdicts = [
        str(claim.get("verdict", "")).lower()
        for claim in claims
    ]

    unsupported_count = verdicts.count("unsupported")
    partial_count = verdicts.count("partial")
    supported_count = verdicts.count("supported")
    safety_count = verdicts.count("allowed_safety")
    control_count = verdicts.count("control")
    prompt_leak_count = verdicts.count("prompt_leak")

    citation_mismatch_count = sum(
        1
        for claim in claims
        if (
            str(claim.get("verdict", "")).lower()
            in {"supported", "partial", "unsupported"}
            and claim.get("citation_match") is False
        )
    )

    if judge_error:
        grounding_verdict = "error"
        passed = False
    elif (
        unsupported_count == 0
        and partial_count == 0
        and prompt_leak_count == 0
        and citation_mismatch_count == 0
        and citation_checks["citation_numbers_valid"]
    ):
        grounding_verdict = "pass"
        passed = True
    else:
        grounding_verdict = "review"
        passed = False

    summary_row = {
        "id": row.get("id", ""),
        "language": language,
        "question": question,
        "detected_handle": (
            matched_product.product_handle
            if matched_product
            else ""
        ),
        "retrieved_sources": len(passages),
        "grounding_verdict": grounding_verdict,
        "passed": passed,
        "supported_claims": supported_count,
        "partial_claims": partial_count,
        "unsupported_claims": unsupported_count,
        "allowed_safety_claims": safety_count,
        "control_claims": control_count,
        "prompt_leak_claims": prompt_leak_count,
        "citation_mismatches": citation_mismatch_count,
        "invalid_citations": ",".join(
            str(value)
            for value in citation_checks["invalid_citations"]
        ),
        "answer": answer,
        "judge_summary": grounding.get("summary", ""),
        "judge_error": judge_error,
    }

    detail = {
        "id": row.get("id", ""),
        "language": language,
        "question": question,
        "answer": answer,
        "matched_product": (
            {
                "product_handle": matched_product.product_handle,
                "product_title": matched_product.product_title,
            }
            if matched_product
            else None
        ),
        "sources": [
            {
                "source_number": index,
                "id": passage["id"],
                "distance": passage["distance"],
                "metadata": passage["metadata"],
                "text": passage["text"],
            }
            for index, passage in enumerate(passages, start=1)
        ],
        "citation_checks": citation_checks,
        "grounding": grounding,
        "raw_judge_response": raw_judge,
        "judge_error": judge_error,
    }

    return summary_row, detail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=Path,
        default=DEFAULT_EVAL_FILE,
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=DEFAULT_CSV_OUTPUT,
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_JSON_OUTPUT,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    args = parser.parse_args()

    eval_df = pd.read_csv(
        args.file,
        keep_default_na=False,
    )

    if args.limit is not None:
        eval_df = eval_df.head(args.limit)

    summary_rows = []
    details = []

    print("\nAnswer grounding evaluation")
    print("=" * 60)

    for position, (_, row) in enumerate(eval_df.iterrows(), start=1):
        print(f"\n[{position}/{len(eval_df)}] {row['question']}")

        summary, detail = evaluate_case(row)
        summary_rows.append(summary)
        details.append(detail)

        print(
            "  verdict="
            f"{summary['grounding_verdict']} "
            f"unsupported={summary['unsupported_claims']} "
            f"partial={summary['partial_claims']} "
            f"citation_mismatches={summary['citation_mismatches']}"
        )

        if not summary["passed"]:
            if summary["judge_error"]:
                print(f"  judge_error: {summary['judge_error']}")

            for claim in detail["grounding"].get("claims", []):
                if (
                    claim.get("verdict") in {"partial", "unsupported", "prompt_leak"}
                    or (
                        claim.get("verdict") in {"supported", "partial", "unsupported"}
                        and claim.get("citation_match") is False
                    )
                ):
                    print(
                        f"  - {claim.get('verdict')}: "
                        f"{claim.get('claim')}"
                    )
                    print(f"    reason: {claim.get('reason')}")

    results = pd.DataFrame(summary_rows)

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)

    results.to_csv(args.csv_output, index=False)

    args.json_output.write_text(
        json.dumps(details, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    pass_rate = results["passed"].mean() if len(results) else 0.0

    print("\n" + "=" * 60)
    print(f"Cases: {len(results)}")
    print(f"Grounded answer pass rate: {pass_rate:.1%}")
    print(
        "Unsupported claims: "
        f"{int(results['unsupported_claims'].sum())}"
    )
    print(
        "Partially supported claims: "
        f"{int(results['partial_claims'].sum())}"
    )
    print(
        "Citation mismatches: "
        f"{int(results['citation_mismatches'].sum())}"
    )
    print(
        "Prompt leaks: "
        f"{int(results['prompt_leak_claims'].sum())}"
    )
    print(
        "Judge errors: "
        f"{int((results['grounding_verdict'] == 'error').sum())}"
    )
    print(f"\nSummary CSV: {args.csv_output}")
    print(f"Detailed JSON: {args.json_output}")


if __name__ == "__main__":
    main()