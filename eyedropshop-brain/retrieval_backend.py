from __future__ import annotations

import os

from retrieval import (
    retrieve_passages as retrieve_passages_chroma,
)
from postgres_retrieval import (
    retrieve_passages_postgres,
)


VALID_RETRIEVAL_BACKENDS = {
    "postgres",
    "chroma",
}


def get_retrieval_backend():
    backend = os.getenv(
        "BRAIN_RETRIEVAL_BACKEND",
        "postgres",
    ).strip().lower()

    if backend not in VALID_RETRIEVAL_BACKENDS:
        raise ValueError(
            "Unsupported retrieval backend: "
            f"{backend!r}"
        )

    return backend


def retrieve_passages(
    question: str,
    language: str,
    number_of_results=5,
    max_distance=0.85,
    audience: str = "internal",
):
    backend = get_retrieval_backend()

    if backend == "postgres":
        return retrieve_passages_postgres(
            question=question,
            language=language,
            number_of_results=number_of_results,
            max_distance=max_distance,
            audience=audience,
        )

    return retrieve_passages_chroma(
        question=question,
        language=language,
        number_of_results=number_of_results,
        max_distance=max_distance,
        audience=audience,
    )
