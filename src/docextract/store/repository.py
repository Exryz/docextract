"""Repository for persisting documents, running review edits, and querying duplicate invoices."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, col, create_engine, select

from docextract.ingest.splitter import IngestResult
from docextract.schemas.document import DocumentKind, ExtractedDocument, Invoice, Receipt
from docextract.scoring.confidence import ConfidenceReport, score_document_confidence
from docextract.store.models import (
    DocumentNotFoundError,
    DocumentRecord,
    DocumentStatus,
    ReviewAuditEntry,
    assert_valid_transition,
)
from docextract.validate.rules import ValidationReport, validate_document


def _parse_extracted_json(raw_json: str, kind_str: str | None) -> ExtractedDocument:
    payload = json.loads(raw_json)
    resolved_kind = kind_str or payload.get("kind", DocumentKind.INVOICE.value)
    if resolved_kind == DocumentKind.RECEIPT.value:
        return Receipt.model_validate(payload)
    return Invoice.model_validate(payload)


def _apply_field_overrides(
    doc: ExtractedDocument,
    overrides: dict[str, Any],
) -> tuple[ExtractedDocument, list[tuple[str, str, str]]]:
    """Apply dotted/indexed overrides (`total`, `vendor.name`, `line_items[0].amount`) to `doc`.

    Returns `(updated_doc, [(field_path, before_str, after_str), ...])`.
    """
    data: dict[str, Any] = json.loads(doc.model_dump_json())
    changes: list[tuple[str, str, str]] = []

    for path, new_val in overrides.items():
        if "[" in path and "]." in path:
            # Indexed list field, e.g. "line_items[0].amount"
            list_key, rest = path.split("[", 1)
            idx_str, child_key = rest.split("].", 1)
            idx = int(idx_str)
            target_list = data.get(list_key, [])
            if 0 <= idx < len(target_list) and isinstance(target_list[idx], dict):
                old_str = str(target_list[idx].get(child_key, ""))
                new_str = str(new_val)
                if old_str != new_str:
                    target_list[idx][child_key] = new_val
                    changes.append((path, old_str, new_str))
        elif "." in path:
            parent_key, child_key = path.split(".", 1)
            parent_obj = data.get(parent_key)
            if isinstance(parent_obj, dict):
                old_str = str(parent_obj.get(child_key, ""))
                new_str = str(new_val)
                if old_str != new_str:
                    parent_obj[child_key] = new_val
                    changes.append((path, old_str, new_str))
        else:
            old_str = str(data.get(path, ""))
            new_str = str(new_val)
            if old_str != new_str:
                data[path] = new_val
                changes.append((path, old_str, new_str))

    updated: ExtractedDocument = (
        Invoice.model_validate(data)
        if isinstance(doc, Invoice)
        else Receipt.model_validate(data)
    )
    return updated, changes


def _summary_fields(doc: ExtractedDocument) -> tuple[str, str, str | None, str, str]:
    if isinstance(doc, Invoice):
        return (
            DocumentKind.INVOICE.value,
            doc.vendor.name,
            doc.invoice_number,
            doc.currency.value,
            str(doc.total),
        )
    assert isinstance(doc, Receipt)
    return (
        DocumentKind.RECEIPT.value,
        doc.merchant.name,
        doc.receipt_number,
        doc.currency.value,
        str(doc.total),
    )


class DocumentRepository:
    """SQLModel persistence layer for document records, state transitions, and reviewer audits."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        SQLModel.metadata.create_all(self.engine)

    @classmethod
    def from_url(cls, database_url: str) -> DocumentRepository:
        if database_url.startswith("sqlite:///") and database_url != "sqlite:///:memory:":
            db_file = Path(database_url.removeprefix("sqlite:///"))
            db_file.parent.mkdir(parents=True, exist_ok=True)
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        engine = create_engine(database_url, connect_args=connect_args)
        return cls(engine)

    def create_from_ingest(
        self,
        ingest: IngestResult,
        filename: str = "upload.pdf",
    ) -> DocumentRecord:
        record = DocumentRecord(
            doc_sha256=ingest.doc_sha256,
            filename=filename,
            mime_type=ingest.mime_type.value,
            page_count=ingest.page_count,
            status=DocumentStatus.RECEIVED.value,
            pages_json=json.dumps([p.model_dump() for p in ingest.pages]),
        )
        with Session(self.engine) as session:
            session.add(record)
            session.commit()
            session.refresh(record)
            return record

    def get(self, doc_id: str) -> DocumentRecord:
        with Session(self.engine) as session:
            rec = session.get(DocumentRecord, doc_id)
            if rec is None:
                raise DocumentNotFoundError(f"Document '{doc_id}' not found")
            return rec

    def get_extracted_document(self, doc_id: str) -> ExtractedDocument | None:
        rec = self.get(doc_id)
        if not rec.extracted_json:
            return None
        return _parse_extracted_json(rec.extracted_json, rec.kind)

    def list_documents(
        self,
        *,
        status: DocumentStatus | str | None = None,
        kind: DocumentKind | str | None = None,
        limit: int = 50,
    ) -> list[DocumentRecord]:
        with Session(self.engine) as session:
            stmt = select(DocumentRecord).order_by(col(DocumentRecord.created_at).desc())
            if status is not None:
                st_val = status.value if isinstance(status, DocumentStatus) else str(status)
                stmt = stmt.where(DocumentRecord.status == st_val)
            if kind is not None:
                k_val = kind.value if isinstance(kind, DocumentKind) else str(kind)
                stmt = stmt.where(DocumentRecord.kind == k_val)
            stmt = stmt.limit(limit)
            return list(session.exec(stmt).all())

    def find_duplicate_invoice(
        self,
        vendor_name: str,
        invoice_number: str,
        *,
        exclude_doc_id: str | None = None,
    ) -> DocumentRecord | None:
        """Check if a non-rejected invoice with the same vendor and invoice_number exists."""
        norm_vendor = vendor_name.strip().lower()
        norm_num = invoice_number.strip().lower()
        with Session(self.engine) as session:
            stmt = select(DocumentRecord).where(
                DocumentRecord.kind == DocumentKind.INVOICE.value,
                DocumentRecord.status != DocumentStatus.REJECTED.value,
            )
            candidates = list(session.exec(stmt).all())

        for cand in candidates:
            if exclude_doc_id and cand.id == exclude_doc_id:
                continue
            if (
                (cand.vendor_name or "").strip().lower() == norm_vendor
                and (cand.document_number or "").strip().lower() == norm_num
            ):
                return cand
        return None

    def save_extraction(
        self,
        doc_id: str,
        document: ExtractedDocument,
        validation: ValidationReport,
        confidence: ConfidenceReport,
    ) -> DocumentRecord:
        """Persist extraction, validation, and confidence outputs and route status automatically."""
        target_status = (
            DocumentStatus.NEEDS_REVIEW
            if confidence.needs_review or not validation.passed
            else DocumentStatus.APPROVED
        )
        kind_str, party_name, doc_num, currency_str, total_str = _summary_fields(document)

        with Session(self.engine) as session:
            rec = session.get(DocumentRecord, doc_id)
            if rec is None:
                raise DocumentNotFoundError(f"Document '{doc_id}' not found")
            assert_valid_transition(DocumentStatus(rec.status), target_status)

            rec.kind = kind_str
            rec.vendor_name = party_name
            rec.document_number = doc_num
            rec.currency = currency_str
            rec.total_amount = total_str
            rec.document_confidence = confidence.document_confidence
            rec.extracted_json = document.model_dump_json()
            rec.validation_json = validation.model_dump_json()
            rec.confidence_json = confidence.model_dump_json()
            rec.status = target_status.value
            rec.error_message = None
            rec.updated_at = datetime.now(UTC)

            session.add(rec)
            session.commit()
            session.refresh(rec)
            return rec

    def mark_failed(self, doc_id: str, error_message: str) -> DocumentRecord:
        with Session(self.engine) as session:
            rec = session.get(DocumentRecord, doc_id)
            if rec is None:
                raise DocumentNotFoundError(f"Document '{doc_id}' not found")
            assert_valid_transition(DocumentStatus(rec.status), DocumentStatus.FAILED)
            rec.status = DocumentStatus.FAILED.value
            rec.error_message = error_message
            rec.updated_at = datetime.now(UTC)
            session.add(rec)
            session.commit()
            session.refresh(rec)
            return rec

    def apply_review(
        self,
        doc_id: str,
        *,
        reviewer: str,
        action: DocumentStatus,
        field_overrides: dict[str, Any] | None = None,
        note: str | None = None,
    ) -> DocumentRecord:
        """Apply human field corrections, re-validate, transition status, and write audit rows."""
        if action not in (
            DocumentStatus.APPROVED,
            DocumentStatus.REJECTED,
            DocumentStatus.NEEDS_REVIEW,
        ):
            raise ValueError(f"Invalid review target status: {action.value}")

        with Session(self.engine) as session:
            rec = session.get(DocumentRecord, doc_id)
            if rec is None:
                raise DocumentNotFoundError(f"Document '{doc_id}' not found")

            current_status = DocumentStatus(rec.status)
            assert_valid_transition(current_status, action)

            if field_overrides and rec.extracted_json:
                current_doc = _parse_extracted_json(rec.extracted_json, rec.kind)
                updated_doc, changes = _apply_field_overrides(current_doc, field_overrides)
                new_val_report = validate_document(updated_doc)
                new_conf_report = score_document_confidence(
                    updated_doc,
                    validation=new_val_report,
                )
                kind_str, party_name, doc_num, currency_str, total_str = _summary_fields(
                    updated_doc
                )

                rec.kind = kind_str
                rec.vendor_name = party_name
                rec.document_number = doc_num
                rec.currency = currency_str
                rec.total_amount = total_str
                rec.document_confidence = new_conf_report.document_confidence
                rec.extracted_json = updated_doc.model_dump_json()
                rec.validation_json = new_val_report.model_dump_json()
                rec.confidence_json = new_conf_report.model_dump_json()

                for field_path, before_str, after_str in changes:
                    session.add(
                        ReviewAuditEntry(
                            document_id=doc_id,
                            reviewer=reviewer,
                            action="edit_field",
                            field_path=field_path,
                            before_value=before_str,
                            after_value=after_str,
                            note=note,
                        )
                    )

            rec.status = action.value
            rec.updated_at = datetime.now(UTC)
            session.add(rec)
            session.add(
                ReviewAuditEntry(
                    document_id=doc_id,
                    reviewer=reviewer,
                    action=action.value,
                    field_path="status",
                    before_value=current_status.value,
                    after_value=action.value,
                    note=note,
                )
            )
            session.commit()
            session.refresh(rec)
            return rec

    def list_audit_entries(self, doc_id: str) -> list[ReviewAuditEntry]:
        with Session(self.engine) as session:
            stmt = (
                select(ReviewAuditEntry)
                .where(ReviewAuditEntry.document_id == doc_id)
                .order_by(col(ReviewAuditEntry.id).asc())
            )
            return list(session.exec(stmt).all())


__all__ = [
    "DocumentRepository",
    "date",
    "Decimal",
]
