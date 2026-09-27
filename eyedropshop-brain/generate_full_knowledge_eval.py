from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import ollama
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PRODUCT_FILE = BASE_DIR / "prototype_data/products/products_30_bilingual.csv"
FAQ_DIR = BASE_DIR / "knowledge/converted/faqs"
POLICY_DIR = BASE_DIR / "knowledge/converted/policies"
TRANSCRIPT_DIR = BASE_DIR / "knowledge/converted/transcripts"
GOVERNANCE_FILE = BASE_DIR / "governance/knowledge_registry.csv"
DEFAULT_OUTPUT = BASE_DIR / "evaluation/full_knowledge_eval.csv"

OLLAMA_MODEL = "qwen3.5:9b"
SCRIPT_VERSION = "2026-08-24-v3"


PRODUCT_QUESTION_TEMPLATES = {
    "en": {
        "description": "What is {title} used for?",
        "benefits_and_details": "What are the benefits of {title}?",
        "usage": "How do I use {title}?",
        "ingredients": "What are the ingredients in {title}?",
        "warnings": "What are the warnings for {title}?",
    },
    "fr": {
        "description": "À quoi sert {title} ?",
        "benefits_and_details": "Quels sont les avantages de {title} ?",
        "usage": "Comment utiliser {title} ?",
        "ingredients": "Quels sont les ingrédients de {title} ?",
        "warnings": "Quels sont les avertissements pour {title} ?",
    },
}


POLICY_QUESTIONS = {
    ("shipping", "en"): [
        "What is your shipping policy?",
        "How long does shipping take?",
        "How much does shipping cost?",
    ],
    ("shipping", "fr"): [
        "Quelle est votre politique de livraison ?",
        "Quels sont les délais de livraison ?",
        "Combien coûte la livraison ?",
    ],
    ("returns", "en"): [
        "What is your return policy?",
        "Can I return an opened product?",
        "Are original shipping fees refunded?",
        "What should I do if I received the wrong item?",
        "What should I do if my order arrived damaged?",
    ],
    ("returns", "fr"): [
        "Quelle est votre politique de retour ?",
        "Puis-je retourner un produit ouvert ?",
        "Les frais de livraison initiaux sont-ils remboursés ?",
        "Que dois-je faire si j'ai reçu le mauvais article ?",
        "Que dois-je faire si ma commande est arrivée endommagée ?",
    ],
}


def _clean(value) -> str:
    if value is None:
        return ""

    value = str(value).strip()

    return (
        ""
        if value.lower() == "nan"
        else value
    )


def _first_existing_column(
    df: pd.DataFrame,
    names: list[str],
) -> str | None:
    lookup = {
        str(column).strip().lower():
        column
        for column in df.columns
    }

    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]

    return None


def _product_sections_available(
    row: pd.Series,
) -> list[str]:
    """
    Determine which searchable product sections
    contain data for this row.
    """
    section_fields = {
        "description": [
            "description",
            "body_html",
        ],
        "benefits_and_details": [
            "benefits",
            "more_details",
            "more details",
            "product_details",
            "product details",
            "features",
        ],
        "usage": [
            "how_to_use",
            "how to use",
            "usage",
            "dosage",
        ],
        "ingredients": [
            "ingredients",
        ],
        "warnings": [
            "warnings",
        ],
    }

    normalized_row = {
        str(key).strip().lower():
        value
        for key, value
        in row.to_dict().items()
    }

    available = []

    for section, fields in (
        section_fields.items()
    ):
        if any(
            _clean(
                normalized_row.get(
                    field.lower()
                )
            )
            for field in fields
        ):
            available.append(section)

    return available


def _language_from_filename(
    path: Path,
    default: str = "en",
) -> str:
    name = path.stem.lower()

    if re.search(
        r"(?:^|[_\-\s])fr(?:$|[_\-\s])",
        name,
    ):
        return "fr"

    if re.search(
        r"(?:^|[_\-\s])en(?:$|[_\-\s])",
        name,
    ):
        return "en"

    return default


def _policy_section_from_filename(
    path: Path,
) -> str | None:
    name = path.stem.lower()

    if (
        "ship" in name
        or "livraison" in name
    ):
        return "shipping"

    if (
        "return" in name
        or "retour" in name
    ):
        return "returns"

    return None


def _source_keys(
    path: Path,
) -> set[str]:
    values = {
        path.name.lower(),
        path.stem.lower(),
        str(path).lower(),
    }

    return {
        value.replace("\\", "/")
        for value in values
    }


def _approved_registry_rows() -> pd.DataFrame:
    if not GOVERNANCE_FILE.exists():
        return pd.DataFrame()

    df = pd.read_csv(
        GOVERNANCE_FILE,
        keep_default_na=False,
    )

    if "approval_status" in df.columns:
        df = df[
            df[
                "approval_status"
            ]
            .astype(str)
            .str.strip()
            .str.lower()
            == "approved"
        ]

    return df


def _registry_matches_source(
    path: Path,
    registry: pd.DataFrame,
) -> bool | None:
    """
    True:
        source appears in approved governance rows

    False:
        governance registry exists, but this source
        does not match an approved row

    None:
        no usable governance registry is available
    """
    if registry.empty:
        return None

    source_like = [
        column
        for column in registry.columns
        if any(
            token
            in str(column).lower()
            for token in (
                "source",
                "file",
                "path",
                "document",
                "title",
            )
        )
    ]

    if not source_like:
        return None

    keys = _source_keys(path)

    for _, row in registry.iterrows():
        for column in source_like:
            value = (
                _clean(
                    row.get(column)
                )
                .lower()
                .replace("\\", "/")
            )

            if not value:
                continue

            value_name = (
                Path(value).name.lower()
            )
            value_stem = (
                Path(value).stem.lower()
            )

            if (
                value in keys
                or value_name in keys
                or value_stem in keys
                or any(
                    key
                    and key in value
                    for key in keys
                )
            ):
                return True

    return False


def _extract_faq_questions(
    text: str,
) -> list[str]:
    """
    Extract explicit FAQ-style questions from a
    converted FAQ text file.
    """
    candidates = []

    for raw_line in text.splitlines():
        line = re.sub(
            r"^\s*(?:[-*•]|\d+[.)])\s*",
            "",
            raw_line,
        ).strip()

        if not line.endswith("?"):
            continue

        if (
            len(line) < 8
            or len(line) > 240
        ):
            continue

        candidates.append(line)

    for match in re.finditer(
        r"(?:(?<=^)|(?<=[.!]\s))"
        r"([^?\n]{8,220}\?)",
        text,
    ):
        question = (
            " ".join(
                match.group(1).split()
            )
            .strip()
        )

        candidates.append(question)

    seen = set()
    unique = []

    for question in candidates:
        key = question.casefold()

        if key in seen:
            continue

        seen.add(key)
        unique.append(question)

    return unique


def _sample_long_text(
    text: str,
    max_chars: int = 18000,
) -> str:
    """
    Keep prompts bounded while sampling the
    beginning, middle, and end of long transcripts.
    """
    text = text.strip()

    if len(text) <= max_chars:
        return text

    part = max_chars // 3

    middle_start = max(
        0,
        len(text) // 2
        - part // 2,
    )

    return (
        text[:part]
        + "\n\n"
        + "[...middle excerpt...]"
        + "\n\n"
        + text[
            middle_start:
            middle_start + part
        ]
        + "\n\n"
        + "[...final excerpt...]"
        + "\n\n"
        + text[-part:]
    )


def _parse_generated_questions(
    raw: str,
    number: int,
) -> list[str]:
    """
    Parse Qwen output robustly.

    Qwen is asked for a JSON array, but if it
    returns malformed JSON this function falls
    back to extracting quoted questions or
    question-like lines instead of crashing.
    """
    raw = raw.strip()

    if raw.startswith("```"):
        raw = re.sub(
            r"^```(?:json)?\s*",
            "",
            raw,
        )
        raw = re.sub(
            r"\s*```$",
            "",
            raw,
        )

    values = None

    # ----------------------------------------------------------
    # 1) Strict JSON.
    # ----------------------------------------------------------
    try:
        values = json.loads(raw)

    except json.JSONDecodeError:
        pass

    # ----------------------------------------------------------
    # 2) Try a JSON-looking array embedded in other text.
    # ----------------------------------------------------------
    if values is None:
        start = raw.find("[")
        end = raw.rfind("]")

        if (
            start != -1
            and end != -1
            and end > start
        ):
            try:
                values = json.loads(
                    raw[start:end + 1]
                )

            except json.JSONDecodeError:
                pass

    # ----------------------------------------------------------
    # 3) Recover quoted question strings from malformed JSON.
    # ----------------------------------------------------------
    if values is None:
        quoted_questions = re.findall(
            r'["“]([^"”\n]*\?)["”]',
            raw,
        )

        if quoted_questions:
            values = quoted_questions

    # ----------------------------------------------------------
    # 4) Recover question-like lines.
    # ----------------------------------------------------------
    if values is None:
        values = []

        for line in raw.splitlines():
            line = re.sub(
                r"^\s*(?:[-*•]|\d+[.)])\s*",
                "",
                line,
            ).strip()

            line = line.strip(
                " \"'[],-"
            )

            if (
                line.endswith("?")
                and len(line) >= 8
            ):
                values.append(line)

    if not isinstance(values, list):
        return []

    questions = []
    seen = set()

    for value in values:
        question = (
            " ".join(
                str(value).split()
            )
            .strip()
        )

        if not question.endswith("?"):
            continue

        if len(question) < 8:
            continue

        key = question.casefold()

        if key in seen:
            continue

        seen.add(key)
        questions.append(question)

        if len(questions) >= number:
            break

    return questions


def _generate_source_questions(
    text: str,
    language: str,
    number: int,
    source_name: str,
) -> list[str]:
    sample = _sample_long_text(text)

    if language == "fr":
        prompt = f"""
À partir UNIQUEMENT du texte source ci-dessous,
génère {number} questions courtes et factuelles
qui peuvent être répondues directement par ce texte.

Contraintes:
- questions éducatives/informatives seulement;
- aucune recommandation médicale personnalisée;
- aucune question demandant quel produit est le meilleur;
- choisis des faits distinctifs du contenu;
- n'utilise aucune connaissance externe;
- retourne UNIQUEMENT un tableau JSON de chaînes.

SOURCE:
{source_name}

TEXTE:
{sample}
"""

    else:
        prompt = f"""
Using ONLY the source text below, generate
{number} short factual questions that can be
answered directly from this text.

Constraints:
- educational/informational questions only;
- no personalized medical-treatment questions;
- no "which product is best" recommendation questions;
- choose distinctive facts from the content;
- use no outside knowledge;
- return ONLY a JSON array of strings.

SOURCE:
{source_name}

TEXT:
{sample}
"""

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            think=False,
            keep_alive="30m",
            options={
                "temperature": 0.0,
                "num_ctx": 8192,
                "num_predict": 300,
            },
        )

    except Exception as exc:
        print(
            "WARNING: question generation failed "
            f"for {source_name}: {exc}"
        )
        return []

    raw = (
        response["message"]["content"]
        .strip()
    )

    questions = (
        _parse_generated_questions(
            raw=raw,
            number=number,
        )
    )

    if len(questions) < number:
        print(
            "WARNING: generated only "
            f"{len(questions)}/{number} "
            f"usable questions for "
            f"{source_name}."
        )

        if not questions:
            print(
                "RAW MODEL OUTPUT:"
            )
            print(raw[:1000])

    return questions


def build_product_cases() -> list[dict]:
    if not PRODUCT_FILE.exists():
        raise FileNotFoundError(
            "Missing product file: "
            f"{PRODUCT_FILE}"
        )

    products = pd.read_csv(
        PRODUCT_FILE,
        keep_default_na=False,
    )

    handle_col = (
        _first_existing_column(
            products,
            [
                "product_handle",
                "handle",
            ],
        )
    )

    title_col = (
        _first_existing_column(
            products,
            [
                "title",
                "product_title",
                "product_name",
            ],
        )
    )

    lang_col = (
        _first_existing_column(
            products,
            [
                "language",
                "lang",
            ],
        )
    )

    if (
        not handle_col
        or not title_col
        or not lang_col
    ):
        raise ValueError(
            "Product CSV must contain "
            "product_handle/handle, title, "
            "and language."
        )

    cases = []
    counter = 1

    for _, row in products.iterrows():
        handle = _clean(
            row.get(handle_col)
        )

        title = _clean(
            row.get(title_col)
        )

        language = (
            _clean(
                row.get(lang_col)
            )
            .lower()
        )

        if (
            not handle
            or not title
            or language
            not in PRODUCT_QUESTION_TEMPLATES
        ):
            continue

        sections = (
            _product_sections_available(
                row
            )
        )

        for section in sections:
            question = (
                PRODUCT_QUESTION_TEMPLATES[
                    language
                ][section]
                .format(
                    title=title
                )
            )

            cases.append({
                "id":
                f"product-{counter:04d}",
                "knowledge_type":
                "product",
                "language":
                language,
                "question":
                question,
                "expected_handle":
                handle,
                "expected_section":
                section,
                "expected_source":
                "",
                "expect_exact_match":
                1,
                "source_approval":
                "product_dataset",
                "notes":
                "",
            })

            counter += 1

    return cases


def build_policy_cases(
    registry: pd.DataFrame,
) -> list[dict]:
    cases = []
    counter = 1

    if not POLICY_DIR.exists():
        return cases

    for path in sorted(
        POLICY_DIR.glob("*.txt")
    ):
        section = (
            _policy_section_from_filename(
                path
            )
        )

        if not section:
            continue

        language = (
            _language_from_filename(
                path
            )
        )

        approval = (
            _registry_matches_source(
                path,
                registry,
            )
        )

        approval_label = (
            "approved"
            if approval is True
            else (
                "not_approved"
                if approval is False
                else "unknown"
            )
        )

        questions = (
            POLICY_QUESTIONS.get(
                (
                    section,
                    language,
                ),
                [],
            )
        )

        for question in questions:
            cases.append({
                "id":
                f"policy-{counter:04d}",
                "knowledge_type":
                "policy",
                "language":
                language,
                "question":
                question,
                "expected_handle":
                "",
                "expected_section":
                section,
                "expected_source":
                path.name,
                "expect_exact_match":
                0,
                "source_approval":
                approval_label,
                "notes":
                "",
            })

            counter += 1

    return cases


def build_faq_cases(
    registry: pd.DataFrame,
    max_per_file: int = 100,
) -> list[dict]:
    cases = []
    counter = 1

    if not FAQ_DIR.exists():
        return cases

    for path in sorted(
        FAQ_DIR.glob("*.txt")
    ):
        text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        language = (
            _language_from_filename(
                path
            )
        )

        approval = (
            _registry_matches_source(
                path,
                registry,
            )
        )

        approval_label = (
            "approved"
            if approval is True
            else (
                "not_approved"
                if approval is False
                else "unknown"
            )
        )

        questions = (
            _extract_faq_questions(
                text
            )
        )

        if not questions:
            questions = (
                _generate_source_questions(
                    text=text,
                    language=language,
                    number=5,
                    source_name=path.name,
                )
            )

        for question in (
            questions[:max_per_file]
        ):
            cases.append({
                "id":
                f"faq-{counter:04d}",
                "knowledge_type":
                "faq",
                "language":
                language,
                "question":
                question,
                "expected_handle":
                "",
                "expected_section":
                "",
                "expected_source":
                path.name,
                "expect_exact_match":
                0,
                "source_approval":
                approval_label,
                "notes":
                "",
            })

            counter += 1

    return cases


def build_transcript_cases(
    registry: pd.DataFrame,
    questions_per_file: int = 2,
) -> list[dict]:
    cases = []
    counter = 1

    if not TRANSCRIPT_DIR.exists():
        return cases

    for path in sorted(
        TRANSCRIPT_DIR.glob("*.txt")
    ):
        text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        if not text.strip():
            print(
                "WARNING: empty transcript "
                f"skipped: {path.name}"
            )
            continue

        language = (
            _language_from_filename(
                path,
                default="en",
            )
        )

        approval = (
            _registry_matches_source(
                path,
                registry,
            )
        )

        approval_label = (
            "approved"
            if approval is True
            else (
                "not_approved"
                if approval is False
                else "unknown"
            )
        )

        questions = (
            _generate_source_questions(
                text=text,
                language=language,
                number=questions_per_file,
                source_name=path.name,
            )
        )

        if not questions:
            print(
                "WARNING: no usable "
                "questions generated for "
                f"{path.name}; continuing."
            )
            continue

        for question in questions:
            cases.append({
                "id":
                f"transcript-{counter:04d}",
                "knowledge_type":
                "transcript",
                "language":
                language,
                "question":
                question,
                "expected_handle":
                "",
                "expected_section":
                "education",
                "expected_source":
                path.name,
                "expect_exact_match":
                0,
                "source_approval":
                approval_label,
                "notes":
                (
                    "Question generated from "
                    "the source text using "
                    "local Qwen."
                ),
            })

            counter += 1

    return cases


def main():
    print(
        "generate_full_knowledge_eval.py "
        f"version: {SCRIPT_VERSION}"
    )

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--transcript-questions",
        type=int,
        default=2,
        help=(
            "Number of generated questions "
            "per transcript."
        ),
    )

    parser.add_argument(
        "--faq-max-per-file",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    registry = (
        _approved_registry_rows()
    )

    cases = []

    cases.extend(
        build_product_cases()
    )

    cases.extend(
        build_policy_cases(
            registry
        )
    )

    cases.extend(
        build_faq_cases(
            registry,
            max_per_file=(
                args.faq_max_per_file
            ),
        )
    )

    cases.extend(
        build_transcript_cases(
            registry,
            questions_per_file=(
                args.transcript_questions
            ),
        )
    )

    df = pd.DataFrame(cases)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        args.output,
        index=False,
    )

    print()
    print(
        "Full knowledge evaluation set"
    )
    print("=" * 70)
    print(
        f"Saved to: {args.output}"
    )
    print(
        f"Total cases: {len(df)}"
    )

    if df.empty:
        print(
            "WARNING: no evaluation "
            "cases were generated."
        )
        return

    print()
    print(
        "Cases by knowledge type:"
    )
    print(
        df[
            "knowledge_type"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Approval labels for "
        "file-based sources:"
    )

    file_based = df[
        df[
            "knowledge_type"
        ]
        != "product"
    ]

    if not file_based.empty:
        print(
            file_based[
                "source_approval"
            ]
            .value_counts()
            .to_string()
        )

    print()
    print(
        "Unique product handles:",
        df.loc[
            df[
                "knowledge_type"
            ]
            == "product",
            "expected_handle",
        ].nunique(),
    )

    print(
        "Unique policy sources:",
        df.loc[
            df[
                "knowledge_type"
            ]
            == "policy",
            "expected_source",
        ].nunique(),
    )

    print(
        "Unique FAQ sources:",
        df.loc[
            df[
                "knowledge_type"
            ]
            == "faq",
            "expected_source",
        ].nunique(),
    )

    print(
        "Unique transcript sources:",
        df.loc[
            df[
                "knowledge_type"
            ]
            == "transcript",
            "expected_source",
        ].nunique(),
    )


if __name__ == "__main__":
    main()
