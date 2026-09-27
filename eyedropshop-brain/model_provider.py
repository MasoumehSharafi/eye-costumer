from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import ollama


DEFAULT_OLLAMA_MODEL = "qwen3.5:9b"
DEFAULT_KEEP_ALIVE = "30m"


@dataclass(frozen=True)
class ModelResponse:
    content: str
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    done_reason: str | None = None


class ModelProvider(Protocol):
    """
    Interface used by the Brain for language-model generation.
    """

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        num_ctx: int = 8192,
        num_predict: int = 700,
        output_schema: dict | None = None,
    ) -> ModelResponse:
        ...


class OllamaModelProvider:
    """
    Local Ollama implementation of the Brain model provider.
    """

    def __init__(
        self,
        model: str | None = None,
        keep_alive: str | None = None,
    ):
        self.client = ollama.Client(host=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
                                    timeout=float(os.getenv("BRAIN_MODEL_TIMEOUT", "90")))
        self.model = (
            model
            or os.getenv(
                "BRAIN_OLLAMA_MODEL",
                DEFAULT_OLLAMA_MODEL,
            )
        )

        self.keep_alive = (
            keep_alive
            or os.getenv(
                "BRAIN_OLLAMA_KEEP_ALIVE",
                DEFAULT_KEEP_ALIVE,
            )
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        num_ctx: int = 8192,
        num_predict: int = 700,
        output_schema: dict | None = None,
    ) -> ModelResponse:
        response = self.client.chat(
            model=self.model,
            messages=messages,
            think=False,
            format=output_schema,
            keep_alive=self.keep_alive,
            options={
                "temperature": temperature,
                "num_ctx": num_ctx,
                "num_predict": num_predict,
            },
        )

        return ModelResponse(
            content=response[
                "message"
            ]["content"].strip(),
            prompt_tokens=response.get(
                "prompt_eval_count"
            ),
            output_tokens=response.get(
                "eval_count"
            ),
            done_reason=response.get(
                "done_reason"
            ),
        )


@lru_cache(maxsize=1)
def get_model_provider() -> ModelProvider:
    """
    Return the configured Brain model provider.

    The prototype currently supports Ollama.
    Additional providers can be added here later.
    """
    provider_name = os.getenv(
        "BRAIN_MODEL_PROVIDER",
        "ollama",
    ).strip().lower()

    if provider_name == "ollama":
        return OllamaModelProvider()

    raise ValueError(
        "Unsupported Brain model provider: "
        f"{provider_name!r}"
    )
