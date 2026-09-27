from __future__ import annotations

import csv
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_LOG_PATH = Path("logs/knowledge_gaps.csv")

FIELDNAMES = [
    "gap_id",
    "first_seen_utc",
    "last_seen_utc",
    "occurrence_count",
    "language",
    "audience",
    "product",
    "topic",
    "question",
    "status",
    "reason",
    "searched_sources",
    "retrieved_sections",
    "proposed_reviewer_question",
    "review_status",
    "reviewer_notes",
]

INSUFFICIENT_MARKERS = {
    "en": (
        "i don't have enough approved information to answer that",
        "i do not have enough approved information to answer that",
        "i don't have enough approved information",
        "i do not have enough approved information",
    ),
    "fr": (
        "je ne dispose pas de suffisamment d'informations approuvées pour répondre",
        "je ne dispose pas de suffisamment d informations approuvees pour repondre",
        "je n'ai pas suffisamment d'informations approuvées",
        "je n ai pas suffisamment d informations approuvees",
    ),
}


def _normalize(text: str) -> str:
    text = (text or "").strip().lower()
    text = (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("œ", "oe")
    )
    text = re.sub(r"\s+", " ", text)
    return text


def _asciiish(text: str) -> str:
    replacements = {
        "é": "e", "è": "e", "ê": "e", "ë": "e",
        "à": "a", "â": "a", "ä": "a",
        "î": "i", "ï": "i",
        "ô": "o", "ö": "o",
        "ù": "u", "û": "u", "ü": "u",
        "ç": "c",
    }
    out = _normalize(text)
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out


def is_insufficient_answer(answer: str, language: str) -> bool:
    """Return True only for the Brain's controlled insufficient-information response."""
    lang = "fr" if str(language).lower().startswith("fr") else "en"
    normalized = _asciiish(answer)

    for marker in INSUFFICIENT_MARKERS[lang]:
        if _asciiish(marker) in normalized:
            return True

    return False


def _infer_topic(question: str, language: str) -> str:
    q = _asciiish(question)

    groups = [
        ("shipping", ("ship", "shipping", "deliver", "delivery", "livraison", "livrez", "exped")),
        ("returns", ("return", "refund", "replacement", "retour", "rembours", "remplacement")),
        ("ingredients", ("ingredient", "steroid", "preservative", "conservateur", "conservation")),
        ("usage", ("how do i use", "how to use", "use every", "usage", "utiliser", "toutes les heures", "vaporiser")),
        ("warnings", ("warning", "allerg", "safe if", "danger", "avertissement", "sur si", "soja")),
        ("pediatric", ("child", "kid", "year-old", "years old", "enfant", "ans")),
        ("price", ("price", "cheapest", "cost", "prix", "moins cher", "cout")),
        ("recommendation", ("which product", "what product", "recommend", "quel produit", "conseille")),
    ]

    for topic, keywords in groups:
        if any(keyword in q for keyword in keywords):
            return topic

    return "general"


def _metadata_from_passage(passage: Any) -> dict[str, Any]:
    if isinstance(passage, dict):
        metadata = passage.get("metadata")
        if isinstance(metadata, dict):
            merged = dict(metadata)
            for key, value in passage.items():
                if key != "metadata" and key not in merged:
                    merged[key] = value
            return merged
        return passage

    metadata = getattr(passage, "metadata", None)
    if isinstance(metadata, dict):
        return metadata

    return {}


def _first_nonempty(metadata: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _summarize_passages(passages: Iterable[Any] | None) -> tuple[str, str, str]:
    sources: list[str] = []
    sections: list[str] = []
    products: list[str] = []

    for passage in passages or []:
        metadata = _metadata_from_passage(passage)

        source = _first_nonempty(
            metadata,
            ("source", "source_name", "document", "filename", "file", "title", "policy"),
        )
        section = _first_nonempty(
            metadata,
            ("section", "section_name", "content_type", "type"),
        )
        product = _first_nonempty(
            metadata,
            ("product_title", "product_name", "handle", "product"),
        )

        # Policy/document passages may contain generic metadata values such as
        # "Shipping Policy" or "Return Policy" in fields that look product-like.
        # These must never populate the product column in a knowledge gap.
        source_norm = _asciiish(source)
        section_norm = _asciiish(section)
        product_norm = _asciiish(product)

        looks_like_policy = (
            "policy" in source_norm
            or "politique" in source_norm
            or section_norm in {"shipping", "returns", "return", "policy"}
            or "shipping policy" in product_norm
            or "return policy" in product_norm
            or "politique" in product_norm
        )

        if looks_like_policy:
            product = ""

        if source and source not in sources:
            sources.append(source)
        if section and section not in sections:
            sections.append(section)
        if product and product not in products:
            products.append(product)

    return (
        " | ".join(sources),
        " | ".join(sections),
        " | ".join(products),
    )


def _make_gap_id(question: str, language: str) -> str:
    raw = f"{language}|{_asciiish(question)}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def _reviewer_question(question: str, language: str) -> str:
    if str(language).lower().startswith("fr"):
        return (
            f'Pouvez-vous fournir une réponse approuvée et sa source pour : '
            f'« {question.strip()} » ?'
        )
    return (
        f'Can you provide an approved answer and source for: '
        f'"{question.strip()}"?'
    )


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDNAMES})


def log_knowledge_gap(
    *,
    question: str,
    language: str,
    answer: str,
    passages: Iterable[Any] | None = None,
    product: str = "",
    audience: str = "customer",
    reason: str = "",
    log_path: str | Path = DEFAULT_LOG_PATH,
) -> bool:
    """
    Log a question only when the final Brain answer is the controlled
    insufficient-information response.

    Repeated identical questions are de-duplicated. Instead of creating a new
    row, occurrence_count and last_seen_utc are updated.

    Returns:
        True if a gap was logged/updated, False if the answer was not an
        insufficient-information response.
    """
    if not is_insufficient_answer(answer, language):
        return False

    path = Path(log_path)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    gap_id = _make_gap_id(question, language)

    searched_sources, retrieved_sections, passage_products = _summarize_passages(passages)
    resolved_product = (product or passage_products or "").strip()

    if not resolved_product:
        try:
            from product_matcher import detect_exact_product

            match = detect_exact_product(question, language)
            if match is not None:
                resolved_product = (
                    getattr(match, "product_title", "")
                    or getattr(match, "product_handle", "")
                    or ""
                ).strip()
        except Exception:
            # Gap logging must never break the customer-facing answer.
            pass

    topic = _infer_topic(question, language)

    if not reason:
        if searched_sources or retrieved_sections:
            reason = (
                "Approved evidence was retrieved, but it did not contain enough "
                "information to answer the question safely."
            )
        else:
            reason = "No approved evidence was available to answer the question safely."

    rows = _read_rows(path)

    for row in rows:
        if row.get("gap_id") == gap_id:
            try:
                count = int(row.get("occurrence_count") or "1")
            except ValueError:
                count = 1

            row["last_seen_utc"] = now
            row["occurrence_count"] = str(count + 1)

            # Preserve reviewer work, but enrich machine-generated context.
            if resolved_product and not row.get("product"):
                row["product"] = resolved_product
            if searched_sources:
                row["searched_sources"] = searched_sources
            if retrieved_sections:
                row["retrieved_sections"] = retrieved_sections
            if reason:
                row["reason"] = reason

            _write_rows(path, rows)
            return True

    rows.append(
        {
            "gap_id": gap_id,
            "first_seen_utc": now,
            "last_seen_utc": now,
            "occurrence_count": "1",
            "language": language,
            "audience": audience,
            "product": resolved_product,
            "topic": topic,
            "question": question.strip(),
            "status": "insufficient_knowledge",
            "reason": reason,
            "searched_sources": searched_sources,
            "retrieved_sections": retrieved_sections,
            "proposed_reviewer_question": _reviewer_question(question, language),
            "review_status": "open",
            "reviewer_notes": "",
        }
    )

    _write_rows(path, rows)
    return True