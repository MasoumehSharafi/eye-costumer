from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from governance_registry import (
    get_governance_record,
    load_registry,
    record_is_indexable,
)


BASE_DIR = Path(__file__).resolve().parent

PRODUCT_FILE = (
    BASE_DIR
    / "prototype_data"
    / "products"
    / "products_30_bilingual.csv"
)

POLICY_FILES = [
    {
        "path": (
            BASE_DIR
            / "knowledge"
            / "converted"
            / "policies"
            / "shipping_ca_en.txt"
        ),
        "language": "en",
        "content_type": "policy",
    },
    {
        "path": (
            BASE_DIR
            / "knowledge"
            / "converted"
            / "policies"
            / "shipping_ca_fr.txt"
        ),
        "language": "fr",
        "content_type": "policy",
    },
    {
        "path": (
            BASE_DIR
            / "knowledge"
            / "converted"
            / "policies"
            / "return_ca_en.txt"
        ),
        "language": "en",
        "content_type": "policy",
    },
    {
        "path": (
            BASE_DIR
            / "knowledge"
            / "converted"
            / "policies"
            / "return_ca_fr.txt"
        ),
        "language": "fr",
        "content_type": "policy",
    },
]

TRANSCRIPT_DIR = (
    BASE_DIR
    / "knowledge"
    / "converted"
    / "transcripts"
)

TRANSCRIPT_FILES = [
    {
        "path": path,
        "language": "en",
        "content_type": "transcript",
    }
    for path in sorted(
        TRANSCRIPT_DIR.glob("*.txt")
    )
]

VALID_LANGUAGES = {
    "en",
    "fr",
}


REQUIRED_PRODUCT_FIELDS = [
    "product_id",
    "product_handle",
    "language",
    "source",
]

RECOMMENDED_PRODUCT_FIELDS = [
    "title",
]


def clean_value(value):
    if pd.isna(value):
        return ""

    return str(value).strip()


def report(
    level,
    check,
    details,
):
    print(
        f"{level:<4} | "
        f"{check:<30} | "
        f"{details}"
    )


def validate_registry():
    rows = load_registry()

    report(
        "PASS",
        "governance registry",
        f"{len(rows)} valid records",
    )

    return rows


def validate_products():
    if not PRODUCT_FILE.exists():
        report(
            "FAIL",
            "product file exists",
            str(PRODUCT_FILE),
        )
        return False

    products = pd.read_csv(
        PRODUCT_FILE
    )

    passed = True

    report(
        "PASS",
        "product file exists",
        str(PRODUCT_FILE),
    )

    missing_columns = [
        field
        for field in REQUIRED_PRODUCT_FIELDS
        if field not in products.columns
    ]

    if missing_columns:
        report(
            "FAIL",
            "required product columns",
            f"missing={missing_columns}",
        )
        return False

    report(
        "PASS",
        "required product columns",
        "all present",
    )

    missing_required = []

    for row_number, row in (
        products.iterrows()
    ):
        for field in (
            REQUIRED_PRODUCT_FIELDS
        ):
            if not clean_value(
                row.get(field)
            ):
                missing_required.append(
                    (
                        row_number + 2,
                        field,
                    )
                )

    if missing_required:
        passed = False

        for (
            row_number,
            field,
        ) in missing_required:
            report(
                "FAIL",
                "required product value",
                (
                    f"row={row_number}, "
                    f"field={field}"
                ),
            )
    else:
        report(
            "PASS",
            "required product values",
            "no missing values",
        )
    # --------------------------------------------------
    # Recommended fields: warn, but do not block ingestion.
    # --------------------------------------------------
    missing_recommended = []

    for row_number, row in products.iterrows():
        for field in RECOMMENDED_PRODUCT_FIELDS:
            if not clean_value(
                row.get(field)
            ):
                missing_recommended.append(
                    (
                        row_number + 2,
                        field,
                        clean_value(
                            row.get("product_handle")
                        ),
                        clean_value(
                            row.get("language")
                        ),
                    )
                )

    if missing_recommended:
        report(
            "WARN",
            "recommended product values",
            (
                f"{len(missing_recommended)} "
                "missing values"
            ),
        )

        for (
            row_number,
            field,
            product_handle,
            language,
        ) in missing_recommended:
            report(
                "WARN",
                "recommended product value",
                (
                    f"row={row_number}, "
                    f"field={field}, "
                    f"product={product_handle}, "
                    f"language={language}"
                ),
            )
    else:
        report(
            "PASS",
            "recommended product values",
            "all present",
        )

    invalid_languages = sorted(
        {
            clean_value(value)
            for value
            in products["language"]
            if clean_value(value)
            not in VALID_LANGUAGES
        }
    )

    if invalid_languages:
        passed = False

        report(
            "FAIL",
            "product languages",
            (
                "invalid="
                f"{invalid_languages}"
            ),
        )
    else:
        report(
            "PASS",
            "product languages",
            "en/fr only",
        )

    duplicate_keys = []

    key_counter = Counter()

    for _, row in products.iterrows():
        key = (
            clean_value(
                row["product_id"]
            ),
            clean_value(
                row["language"]
            ),
        )

        key_counter[key] += 1

    for key, count in (
        key_counter.items()
    ):
        if count > 1:
            duplicate_keys.append(
                (
                    key,
                    count,
                )
            )

    if duplicate_keys:
        passed = False

        for key, count in duplicate_keys:
            report(
                "FAIL",
                "duplicate product record",
                (
                    f"key={key}, "
                    f"count={count}"
                ),
            )
    else:
        report(
            "PASS",
            "duplicate product records",
            "none",
        )

    handle_languages = defaultdict(
        set
    )

    for _, row in products.iterrows():
        handle = clean_value(
            row["product_handle"]
        )

        language = clean_value(
            row["language"]
        )

        if handle and language:
            handle_languages[
                handle
            ].add(language)

    incomplete_bilingual = sorted(
        handle
        for handle, languages
        in handle_languages.items()
        if languages != {
            "en",
            "fr",
        }
    )

    if incomplete_bilingual:
        report(
            "WARN",
            "bilingual product coverage",
            (
                "missing EN/FR pair for "
                f"{incomplete_bilingual}"
            ),
        )
    else:
        report(
            "PASS",
            "bilingual product coverage",
            (
                "all product handles "
                "have EN + FR"
            ),
        )

    governance_failures = 0

    checked_sources = set()

    for _, row in products.iterrows():
        source = clean_value(
            row["source"]
        )

        language = clean_value(
            row["language"]
        )

        key = (
            source,
            language,
        )

        if key in checked_sources:
            continue

        checked_sources.add(key)

        try:
            record = (
                get_governance_record(
                    source=source,
                    content_type=(
                        "product"
                    ),
                    language=language,
                    country="CA",
                )
            )
        except Exception as exc:
            governance_failures += 1

            report(
                "FAIL",
                "product governance",
                (
                    f"{source} / "
                    f"{language}: {exc}"
                ),
            )

            continue

        if not record_is_indexable(
            record
        ):
            governance_failures += 1

            report(
                "FAIL",
                "product governance",
                (
                    f"{source} / "
                    f"{language} "
                    "is not indexable"
                ),
            )

    if governance_failures == 0:
        report(
            "PASS",
            "product governance",
            (
                f"{len(checked_sources)} "
                "source/language pairs "
                "approved"
            ),
        )
    else:
        passed = False

    return passed


def validate_policies():
    passed = True

    seen_sources = set()

    for policy in POLICY_FILES:
        path = policy["path"]

        if not path.exists():
            passed = False

            report(
                "FAIL",
                "policy file exists",
                str(path),
            )

            continue

        report(
            "PASS",
            "policy file exists",
            path.name,
        )

        source = path.name

        if source in seen_sources:
            passed = False

            report(
                "FAIL",
                "duplicate policy source",
                source,
            )

        seen_sources.add(
            source
        )

        text = path.read_text(
            encoding="utf-8",
            errors="replace",
        ).strip()

        if not text:
            passed = False

            report(
                "FAIL",
                "policy content",
                (
                    f"{source} is empty"
                ),
            )
        else:
            report(
                "PASS",
                "policy content",
                (
                    f"{source}: "
                    f"{len(text)} chars"
                ),
            )

        try:
            record = (
                get_governance_record(
                    source=source,
                    content_type=(
                        policy[
                            "content_type"
                        ]
                    ),
                    language=(
                        policy[
                            "language"
                        ]
                    ),
                    country="CA",
                )
            )
        except Exception as exc:
            passed = False

            report(
                "FAIL",
                "policy governance",
                (
                    f"{source}: {exc}"
                ),
            )

            continue

        if not record_is_indexable(
            record
        ):
            passed = False

            report(
                "FAIL",
                "policy governance",
                (
                    f"{source} "
                    "is not indexable"
                ),
            )
        else:
            report(
                "PASS",
                "policy governance",
                (
                    f"{source}: approved"
                ),
            )

    return passed


def validate_transcripts():
    passed = True

    if not TRANSCRIPT_FILES:
        report(
            "FAIL",
            "transcript files",
            "no transcript files found",
        )
        return False

    seen_sources = set()

    for transcript in TRANSCRIPT_FILES:
        path = transcript["path"]

        if not path.exists():
            passed = False

            report(
                "FAIL",
                "transcript file exists",
                str(path),
            )

            continue

        source = path.name

        report(
            "PASS",
            "transcript file exists",
            source,
        )

        if source in seen_sources:
            passed = False

            report(
                "FAIL",
                "duplicate transcript source",
                source,
            )

        seen_sources.add(
            source
        )

        text = path.read_text(
            encoding="utf-8",
            errors="replace",
        ).strip()

        if not text:
            passed = False

            report(
                "FAIL",
                "transcript content",
                f"{source} is empty",
            )
        else:
            report(
                "PASS",
                "transcript content",
                (
                    f"{source}: "
                    f"{len(text)} chars"
                ),
            )

        try:
            record = (
                get_governance_record(
                    source=source,
                    content_type=(
                        transcript[
                            "content_type"
                        ]
                    ),
                    language=(
                        transcript[
                            "language"
                        ]
                    ),
                    country="CA",
                )
            )
        except Exception as exc:
            passed = False

            report(
                "FAIL",
                "transcript governance",
                (
                    f"{source}: {exc}"
                ),
            )

            continue

        if not record_is_indexable(
            record
        ):
            passed = False

            report(
                "FAIL",
                "transcript governance",
                (
                    f"{source} "
                    "is not indexable"
                ),
            )
        else:
            report(
                "PASS",
                "transcript governance",
                (
                    f"{source}: approved"
                ),
            )

    return passed

def main():
    print(
        "\nEyeDropShop ingestion validation"
    )
    print(
        "=" * 70
    )

    overall_passed = True

    try:
        validate_registry()
    except Exception as exc:
        report(
            "FAIL",
            "governance registry",
            str(exc),
        )

        overall_passed = False

    print(
        "\n--- Product validation ---"
    )

    if not validate_products():
        overall_passed = False

    print(
        "\n--- Policy validation ---"
    )

    if not validate_policies():
        overall_passed = False

    print(
        "\n--- Transcript validation ---"
    )

    if not validate_transcripts():
        overall_passed = False

    print(
        "\n" + "=" * 70
    )

    if overall_passed:
        print(
            "PASS: ingestion sources are "
            "ready for indexing."
        )
    else:
        print(
            "FAIL: ingestion validation "
            "found blocking problems."
        )

        raise SystemExit(1)


if __name__ == "__main__":
    main()
