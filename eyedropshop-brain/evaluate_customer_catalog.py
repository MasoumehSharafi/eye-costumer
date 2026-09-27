"""Exhaustively verify customer-facing edsAI retrieval for every catalog variant.

This evaluator deliberately bypasses answer generation. It verifies the factual
boundary beneath the conversational LLM: product matching and section retrieval
for every live edsAI product/variant. Natural-language robustness is covered by
the separate live query matrix and offline regressions.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import time

import edsai_retrieval as retrieval
from edsai_client import clean_metafields, get_all_inventory, get_all_products


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "evaluation/customer_catalog_audit.json"
SECTIONS = (
    "description",
    "price",
    "inventory",
    "ingredients",
    "warnings",
    "usage",
    "benefits_and_details",
    "product_details",
)


def expected_sections(product: dict, inventory_ids: set[str]) -> set[str]:
    metafields = clean_metafields(product.get("metafields", []))
    expected = {"description"}

    if product.get("retailPrice") is not None:
        expected.add("price")
    if product.get("id") in inventory_ids:
        expected.add("inventory")
    if metafields.get("ingredients"):
        expected.add("ingredients")
    if metafields.get("warnings"):
        expected.add("warnings")
    if metafields.get("how_to_use") or metafields.get("dosage"):
        expected.add("usage")
    if any(
        metafields.get(key)
        for key in (
            "accordion_benefits",
            "more_details",
            "product_details",
            "features",
            "guarantee",
        )
    ):
        expected.add("benefits_and_details")
    if metafields.get("expiry_date"):
        expected.add("product_details")

    return expected


def run_audit() -> dict:
    products = get_all_products()
    inventory = get_all_inventory()

    # Use the exact snapshot for every check. This prevents repeated API reads and
    # ensures inventory and product assertions refer to the same evaluation run.
    retrieval._product_cache = products
    retrieval._product_cache_time = time.time()
    retrieval._inventory_cache = inventory
    retrieval._inventory_cache_time = time.time()

    inventory_ids = {
        row.get("variantId") for row in inventory if row.get("variantId")
    }
    failures: list[dict] = []
    section_present = Counter()
    section_missing = Counter()

    for product in products:
        expected = expected_sections(product, inventory_ids)
        variant_id = product.get("id")
        title = product.get("title") or ""

        for section in SECTIONS:
            passages = retrieval._product_passages(product, {section}, "en")
            matching = [
                passage
                for passage in passages
                if passage.get("metadata", {}).get("section") == section
            ]
            should_exist = section in expected

            if should_exist:
                section_present[section] += 1
            else:
                section_missing[section] += 1

            if bool(matching) != should_exist:
                failures.append(
                    {
                        "kind": "section_presence",
                        "title": title,
                        "variant": product.get("variantTitle"),
                        "variant_id": variant_id,
                        "section": section,
                        "expected": should_exist,
                        "actual_passages": len(matching),
                    }
                )
                continue

            for passage in matching:
                metadata = passage.get("metadata", {})
                if (
                    metadata.get("source") != "edsAI"
                    or metadata.get("variant_id") != variant_id
                    or not str(passage.get("text", "")).strip()
                ):
                    failures.append(
                        {
                            "kind": "passage_contract",
                            "title": title,
                            "variant": product.get("variantTitle"),
                            "variant_id": variant_id,
                            "section": section,
                        }
                    )

                if section == "price":
                    # A source-provided variant title may itself contain a symbol
                    # (for example a "$25" gift-card option). Check only the
                    # retail-price claim, where currency must remain unspecified.
                    price_claim = passage.get("text", "").split(
                        "Retail price:", 1
                    )[-1].split(".", 1)[0]
                    if any(
                        token in price_claim
                        for token in ("$", "CAD", "USD", "EUR", "GBP")
                    ):
                        failures.append(
                            {
                                "kind": "invented_currency",
                                "title": title,
                                "variant": product.get("variantTitle"),
                                "variant_id": variant_id,
                            }
                        )

                if section == "product_details":
                    variant_title = product.get("variantTitle")
                    if (
                        variant_title
                        and variant_title != "Default Title"
                        and variant_title not in passage.get("text", "")
                    ):
                        failures.append(
                            {
                                "kind": "missing_variant_label",
                                "title": title,
                                "variant": variant_title,
                                "variant_id": variant_id,
                            }
                        )

    unique_titles = sorted({product.get("title") or "" for product in products})
    for title in unique_titles:
        matches = retrieval.find_products_in_question(title)
        normalized_title = retrieval._normalize(title)
        if not any(
            retrieval._normalize(product.get("title")) == normalized_title
            for product in matches
        ):
            failures.append({"kind": "title_matching", "title": title})

    return {
        "passed": not failures,
        "evaluated_at_unix": time.time(),
        "product_count": len({p.get("productId") for p in products}),
        "unique_title_count": len(unique_titles),
        "variant_count": len(products),
        "inventory_row_count": len(inventory),
        "section_checks": len(products) * len(SECTIONS),
        "title_matching_checks": len(unique_titles),
        "section_present": dict(section_present),
        "section_missing": dict(section_missing),
        "failure_count": len(failures),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    result = run_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in result.items() if key != "failures"}))
    if result["failures"]:
        print(json.dumps(result["failures"][:20], ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
