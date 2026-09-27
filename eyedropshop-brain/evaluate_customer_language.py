"""Live semantic-planning evaluation for wording, typos, and safety language."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import time

from edsai_intelligence import understand_query


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "evaluation/customer_language_audit.json"


@dataclass(frozen=True)
class Case:
    question: str
    language: str
    intents: tuple[str, ...]
    sections: tuple[str, ...] = ()
    personal: bool = False
    high_risk: bool = False
    named_product: bool = True


FACT_CASES = (
    # Overview
    Case("what is LUMIFY", "en", ("product_fact",), ("description",)),
    Case("wht is lumify", "en", ("product_fact",), ("description",)),
    Case("tell me about CALMO Eye Spray", "en", ("product_fact",), ("description",)),
    Case("CALMO Eye Spray?", "en", ("product_fact",), ("description",)),
    # Price
    Case("how much is CALMO Eye Spray", "en", ("product_fact",), ("price",)),
    Case("hoe much is CALMO Eye Spray", "en", ("product_fact",), ("price",)),
    Case("wats the price of LUMIFY", "en", ("product_fact",), ("price",)),
    Case("what would LUMIFY set me back", "en", ("product_fact",), ("price",)),
    Case("cost for Eye Allergy Relief Kit?", "en", ("product_fact",), ("price",)),
    # Inventory
    Case("is LUMIFY available", "en", ("product_fact",), ("inventory",)),
    Case("is lumify avalable", "en", ("product_fact",), ("inventory",)),
    Case("do you have CALMO Eye Spray right now", "en", ("product_fact",), ("inventory",)),
    Case("Eye Allergy Relief Kit in stok?", "en", ("product_fact",), ("inventory",)),
    # Ingredients / preservatives
    Case("what are the ingredients in LUMIFY", "en", ("product_fact",), ("ingredients",)),
    Case("whats in CALMO Eye Spray", "en", ("product_fact",), ("ingredients",)),
    Case("ingredints in lumify?", "en", ("product_fact",), ("ingredients",)),
    Case("does lumify have perservatives", "en", ("product_fact",), ("ingredients", "description", "benefits_and_details")),
    # Warnings
    Case("any warnings for LUMIFY", "en", ("product_fact",), ("warnings",)),
    Case("any warnigns for lumify", "en", ("product_fact",), ("warnings",)),
    Case("what should I know before using CALMO Eye Spray", "en", ("product_fact",), ("warnings",)),
    Case("contraindications for LUMIFY", "en", ("product_fact",), ("warnings",)),
    # Usage
    Case("how do I use LUMIFY", "en", ("product_fact",), ("usage",)),
    Case("how offten can i use lumify", "en", ("product_fact",), ("usage",)),
    Case("directions for CALMO Eye Spray", "en", ("product_fact",), ("usage",)),
    Case("how many times per day can I apply LUMIFY", "en", ("product_fact",), ("usage",)),
    # Expiry
    Case("when does LUMIFY expire", "en", ("product_fact",), ("product_details",)),
    Case("what is the expiry date for LUMIFY", "en", ("product_fact",), ("product_details",)),
    Case("expirty date of CALMO Eye Spray", "en", ("product_fact",), ("product_details",)),
    Case("how long before LUMIFY expires", "en", ("product_fact",), ("product_details",)),
    # Benefits/details and multi-field requests
    Case("what are the benefits of CALMO Eye Spray", "en", ("product_fact",), ("benefits_and_details",)),
    Case("features of LUMIFY", "en", ("product_fact",), ("benefits_and_details",)),
    Case("is CALMO available and what does it cost", "en", ("product_fact",), ("inventory", "price")),
    Case("price ingredients and warnings for LUMIFY", "en", ("product_fact",), ("price", "ingredients", "warnings")),
    # French
    Case("combien coûte CALMO Eye Spray", "fr", ("product_fact",), ("price",)),
    Case("est-ce que LUMIFY est disponible", "fr", ("product_fact",), ("inventory",)),
    Case("quels sont les ingrédients de LUMIFY", "fr", ("product_fact",), ("ingredients",)),
    Case("comment utiliser LUMIFY", "fr", ("product_fact",), ("usage",)),
    Case("quelles sont les mises en garde de LUMIFY", "fr", ("product_fact",), ("warnings",)),
)


DISCOVERY_CASES = (
    Case("my eyes are itchy what can I use", "en", ("symptom_discovery",), personal=True, named_product=False),
    Case("my eyes r itchy wat drops can i use", "en", ("symptom_discovery",), personal=True, named_product=False),
    Case("my eyes are red what should I get", "en", ("symptom_discovery",), personal=True, named_product=False),
    Case("anything for red eyes", "en", ("product_search",), named_product=False),
    Case("something preservative free for dry eyes", "en", ("product_search",), named_product=False),
    Case("I need drops I can use with contacts", "en", ("product_search",), named_product=False),
    Case("my eye hurts what drops should I use", "en", ("symptom_discovery",), personal=True, high_risk=True, named_product=False),
    Case("sudden blurry vision and eye pain what can I use", "en", ("symptom_discovery",), personal=True, high_risk=True, named_product=False),
    Case("i got cleaner in my eye what should i buy", "en", ("symptom_discovery",), personal=True, high_risk=True, named_product=False),
    Case("my eye has pus and severe swelling what can I use", "en", ("symptom_discovery",), personal=True, high_risk=True, named_product=False),
)


def run() -> dict:
    details = []
    cases = FACT_CASES + DISCOVERY_CASES

    for index, case in enumerate(cases, 1):
        plan = understand_query(case.question, case.language)
        actual_sections = set(plan.get("requested_sections", []))
        issues = []

        if plan.get("intent") not in case.intents:
            issues.append(f"intent={plan.get('intent')!r}")
        missing = set(case.sections) - actual_sections
        if missing:
            issues.append(f"missing_sections={sorted(missing)!r}")
        if bool(plan.get("personal_symptom")) != case.personal:
            issues.append(f"personal_symptom={plan.get('personal_symptom')!r}")
        if bool(plan.get("high_risk")) != case.high_risk:
            issues.append(f"high_risk={plan.get('high_risk')!r}")
        if bool(plan.get("named_product_query")) != case.named_product:
            issues.append(f"named_product_query={plan.get('named_product_query')!r}")

        details.append(
            {
                **asdict(case),
                "passed": not issues,
                "issues": issues,
                "actual": {
                    "intent": plan.get("intent"),
                    "sections": sorted(actual_sections),
                    "personal_symptom": plan.get("personal_symptom"),
                    "high_risk": plan.get("high_risk"),
                    "named_product_query": plan.get("named_product_query"),
                },
            }
        )
        if index % 5 == 0 or issues:
            print(f"completed={index}/{len(cases)} failures={sum(not d['passed'] for d in details)}", flush=True)

    failures = [detail for detail in details if not detail["passed"]]
    return {
        "passed": not failures,
        "evaluated_at_unix": time.time(),
        "case_count": len(cases),
        "fact_case_count": len(FACT_CASES),
        "discovery_safety_case_count": len(DISCOVERY_CASES),
        "failure_count": len(failures),
        "details": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "details"}))
    failures = [detail for detail in result["details"] if not detail["passed"]]
    if failures:
        print(json.dumps(failures, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
