from __future__ import annotations

import logging
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from answer_generator import _audited_response, generate_response
from auth import authenticate_request
from conversation import plan_turn
from edsai_retrieval import (
    merge_with_existing_passages,
    retrieve_edsai_passages,
)
from retrieval_backend import retrieve_passages
from schemas import BrainResponse


app = FastAPI(
    title="EyeDropShop Brain API",
    version="1.1",
    description="Internal API for the EyeDropShop Brain foundation prototype.",
)


class HistoryMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(
        min_length=1,
        max_length=6000,
    )


class AnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        min_length=1,
        max_length=2000,
        description="Question to ask the EyeDropShop Brain.",
    )

    language: Literal["en", "fr"] = Field(
        default="en",
        description="Response and retrieval language.",
    )

    history: list[HistoryMessage] = Field(
        default_factory=list,
        max_length=8,
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "eyedropshop-brain",
        "version": "1.1",
    }


@app.post(
    "/v1/answer",
    response_model=BrainResponse,
)
def answer_question(
    request: AnswerRequest,
    audience: str = Depends(authenticate_request),
):
    try:
        turn = plan_turn(
            request.question,
            [message.model_dump() for message in request.history],
            request.language,
        )

        question = turn["question"]
        language = turn["language"]

        if turn.get("reply") or turn.get("clarification"):
            return _audited_response(
                question=request.question,
                status=(
                    "answerable"
                    if turn.get("reply")
                    else "insufficient_knowledge"
                ),
                answer=(
                    turn.get("reply")
                    or turn["clarification"]
                ),
                language=language,
                passages=[],
                audience=audience,
            )

        if audience == "customer":
            # Customer Agent:
            # edsAI is the sole authoritative source for product/business facts.
            passages = retrieve_edsai_passages(
                question=question,
                language=language,
            )
        else:
            # Doctor/Internal:
            # governed Brain knowledge may be combined with edsAI.
            passages, _ = retrieve_passages(
                question=question,
                language=language,
                audience=audience,
            )

            edsai_passages = retrieve_edsai_passages(
                question=question,
                language=language,
            )

            passages = merge_with_existing_passages(
                passages,
                edsai_passages,
            )

        return generate_response(
            question=question,
            language=language,
            passages=passages,
            audience=audience,
        )

    except HTTPException:
        raise

    except Exception:
        logging.getLogger(__name__).warning(
            "Brain dependency unavailable",
            exc_info=True,
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "The assistant is temporarily unavailable. "
                "Please try again."
            ),
        )
