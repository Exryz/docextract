"""SQLModel database tables, document state machine, and audit log models."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum

from sqlmodel import Field, SQLModel


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


class DocumentStatus(StrEnum):
    RECEIVED = "received"
    EXTRACTED = "extracted"
    NEEDS_REVIEW = "needs_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    FAILED = "failed"


ALLOWED_TRANSITIONS: dict[DocumentStatus, frozenset[DocumentStatus]] = {
    DocumentStatus.RECEIVED: frozenset(
        {
            DocumentStatus.EXTRACTED,
            DocumentStatus.NEEDS_REVIEW,
            DocumentStatus.APPROVED,
            DocumentStatus.FAILED,
        }
    ),
    DocumentStatus.EXTRACTED: frozenset(
        {
            DocumentStatus.NEEDS_REVIEW,
            DocumentStatus.APPROVED,
            DocumentStatus.REJECTED,
        }
    ),
    DocumentStatus.NEEDS_REVIEW: frozenset(
        {
            DocumentStatus.APPROVED,
            DocumentStatus.REJECTED,
        }
    ),
    DocumentStatus.APPROVED: frozenset({DocumentStatus.NEEDS_REVIEW}),
    DocumentStatus.REJECTED: frozenset({DocumentStatus.NEEDS_REVIEW}),
    DocumentStatus.FAILED: frozenset({DocumentStatus.RECEIVED}),
}


class InvalidStateTransitionError(ValueError):
    """Raised when attempting an illegal `DocumentStatus` transition."""


class DocumentNotFoundError(KeyError):
    """Raised when a requested document ID does not exist in the repository."""


def assert_valid_transition(current: DocumentStatus, target: DocumentStatus) -> None:
    if current == target:
        return
    allowed = ALLOWED_TRANSITIONS.get(current, frozenset())
    if target not in allowed:
        raise InvalidStateTransitionError(
            f"Cannot transition document from '{current.value}' to '{target.value}'"
        )


class DocumentRecord(SQLModel, table=True):
    """Persisted record for an ingested and extracted invoice or receipt."""

    __tablename__ = "document_record"

    id: str = Field(default_factory=_new_id, primary_key=True)
    doc_sha256: str = Field(index=True)
    filename: str = Field(default="upload.pdf")
    mime_type: str = Field(default="application/pdf")
    page_count: int = Field(default=1)
    status: str = Field(default=DocumentStatus.RECEIVED.value, index=True)

    kind: str | None = Field(default=None, index=True)
    vendor_name: str | None = Field(default=None, index=True)
    document_number: str | None = Field(default=None, index=True)
    currency: str | None = Field(default=None)
    total_amount: str | None = Field(default=None)
    document_confidence: float | None = Field(default=None)

    extracted_json: str | None = Field(default=None)
    validation_json: str | None = Field(default=None)
    confidence_json: str | None = Field(default=None)
    pages_json: str | None = Field(default=None)
    error_message: str | None = Field(default=None)

    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)


class ReviewAuditEntry(SQLModel, table=True):
    """Immutable audit log row recording a human reviewer action or field edit."""

    __tablename__ = "review_audit_entry"

    id: int | None = Field(default=None, primary_key=True)
    document_id: str = Field(index=True)
    reviewer: str
    action: str = Field(description="e.g. 'edit_field', 'approve', 'reject'")
    field_path: str | None = Field(default=None)
    before_value: str | None = Field(default=None)
    after_value: str | None = Field(default=None)
    note: str | None = Field(default=None)
    created_at: datetime = Field(default_factory=_utc_now)
