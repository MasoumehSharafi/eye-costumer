from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


Language = Literal[
    "en",
    "fr",
]

Audience = Literal[
    "internal",
    "doctor",
    "customer",
    "shared",
]

ApprovalStatus = Literal[
    "draft",
    "approved",
    "expired",
    "rejected",
]

BrainStatus = Literal[
    "answerable",
    "insufficient_knowledge",
    "restricted",
    "conflicting_sources",
    "human_review_required",
]

ReviewStatus = Literal[
    "open",
    "in_review",
    "resolved",
    "dismissed",
]


class ProductRecord(BaseModel):
    product_id: str
    product_handle: str
    title: str | None = None
    language: Language
    country: str = "CA"
    source: str
    url: str | None = None


class GovernanceRecord(BaseModel):
    registry_id: str
    source: str
    content_type: str
    language: Language
    country: str = "CA"

    audience: Audience
    approval_status: ApprovalStatus

    approver: str | None = None
    knowledge_version: str
    review_date: str | None = None
    expires_at: str | None = None
    notes: str | None = None


class KnowledgePassage(BaseModel):
    source_id: str | None = None
    source_title: str | None = None
    section: str | None = None
    text: str = ""
    distance: float | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class Citation(BaseModel):
    label: str
    source_id: str | None = None
    source_title: str | None = None
    section: str | None = None
    url: str | None = None


class AppliedPermissions(BaseModel):
    audience: str
    language: str
    knowledge_scope: str


class ReviewRequirement(BaseModel):
    required: bool
    reason: str | None = None


class BrainResponse(BaseModel):
    status: BrainStatus
    answer: str

    evidence: list[KnowledgePassage]
    citations: list[Citation]

    applied_permissions: AppliedPermissions
    review: ReviewRequirement

    trace_id: str


class KnowledgeGap(BaseModel):
    gap_id: str
    question: str
    language: Language
    audience: str

    product: str | None = None
    topic: str | None = None

    status: str
    reason: str

    searched_sources: str | None = None
    retrieved_sections: str | None = None

    proposed_reviewer_question: str | None = None

    review_status: str = "open"
    reviewer_notes: str | None = None


class ReviewItem(BaseModel):
    review_id: str
    trace_id: str

    question: str
    language: Language

    status: BrainStatus
    reason: str

    review_status: ReviewStatus

    reviewer: str | None = None
    reviewer_notes: str | None = None
    resolution: str | None = None
