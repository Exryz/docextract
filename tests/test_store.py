"""Unit tests for `docextract.store`: CRUD, state transitions, duplicates, and audit trail."""

from __future__ import annotations

import pytest

from datagen import FaultType, LayoutStyle, generate_sample
from docextract.config import Settings
from docextract.extract import ground_extracted_document
from docextract.ingest import ingest_bytes
from docextract.schemas import Invoice
from docextract.scoring import score_document_confidence
from docextract.store import (
    DocumentNotFoundError,
    DocumentRepository,
    DocumentStatus,
    InvalidStateTransitionError,
)
from docextract.validate import validate_document
from docextract.vision import extract_pdf_text_layer


def test_store_auto_approval_and_duplicate_invoice_detection() -> None:
    repo = DocumentRepository.from_url("sqlite:///:memory:")
    sample = generate_sample(seed=601, layout=LayoutStyle.CORPORATE_TABLE, fault=FaultType.NONE)
    assert isinstance(sample.gold_document, Invoice)

    ingest_res, _ = ingest_bytes(
        sample.pdf_bytes,
        settings=Settings(target_dpi=100),
        persist=False,
    )
    rec = repo.create_from_ingest(ingest_res, filename="inv_601.pdf")
    assert rec.status == DocumentStatus.RECEIVED.value

    layouts = extract_pdf_text_layer(sample.pdf_bytes)
    grounded = ground_extracted_document(sample.gold_document, layouts)
    val_report = validate_document(grounded)
    conf_report = score_document_confidence(grounded, layouts=layouts, validation=val_report)

    saved = repo.save_extraction(rec.id, grounded, val_report, conf_report)
    assert saved.status == DocumentStatus.APPROVED.value
    assert saved.vendor_name == sample.gold_document.vendor.name
    assert saved.document_number == sample.gold_document.invoice_number
    assert saved.total_amount == str(sample.gold_document.total)

    # Round-trip deserialization
    loaded_doc = repo.get_extracted_document(rec.id)
    assert loaded_doc == grounded

    # Duplicate detection finds the existing invoice for the same vendor + invoice_number
    dup = repo.find_duplicate_invoice(
        sample.gold_document.vendor.name,
        sample.gold_document.invoice_number,
    )
    assert dup is not None
    assert dup.id == rec.id
    # Excluding self returns None
    assert (
        repo.find_duplicate_invoice(
            sample.gold_document.vendor.name,
            sample.gold_document.invoice_number,
            exclude_doc_id=rec.id,
        )
        is None
    )


def test_store_review_queue_field_correction_and_audit_trail() -> None:
    repo = DocumentRepository.from_url("sqlite:///:memory:")
    sample = generate_sample(
        seed=602,
        layout=LayoutStyle.CORPORATE_TABLE,
        fault=FaultType.WRONG_TOTAL,
    )
    assert isinstance(sample.gold_document, Invoice)
    assert isinstance(sample.injected_document, Invoice)

    ingest_res, _ = ingest_bytes(
        sample.pdf_bytes,
        settings=Settings(target_dpi=100),
        persist=False,
    )
    rec = repo.create_from_ingest(ingest_res, filename="inv_602_faulty.pdf")

    layouts = extract_pdf_text_layer(sample.pdf_bytes)
    grounded = ground_extracted_document(sample.injected_document, layouts)
    val_report = validate_document(grounded)
    conf_report = score_document_confidence(grounded, layouts=layouts, validation=val_report)

    # Faulty total routes document to `needs_review`
    saved = repo.save_extraction(rec.id, grounded, val_report, conf_report)
    assert saved.status == DocumentStatus.NEEDS_REVIEW.value

    needs_review_list = repo.list_documents(status=DocumentStatus.NEEDS_REVIEW)
    assert len(needs_review_list) == 1
    assert needs_review_list[0].id == rec.id

    # Human reviewer corrects `total` and `vendor.name` and approves the document
    old_total = str(sample.injected_document.total)
    correct_total = str(sample.gold_document.total)
    reviewed = repo.apply_review(
        rec.id,
        reviewer="paul.d",
        action=DocumentStatus.APPROVED,
        field_overrides={
            "total": correct_total,
            "vendor.name": "Northwind Verified Corp",
        },
        note="Corrected grand total typo against line item sum",
    )
    assert reviewed.status == DocumentStatus.APPROVED.value
    assert reviewed.total_amount == correct_total
    assert reviewed.vendor_name == "Northwind Verified Corp"

    # Verify audit trail captured both field edits and the status approval transition
    audits = repo.list_audit_entries(rec.id)
    assert len(audits) == 3

    total_edit = next(a for a in audits if a.field_path == "total")
    assert total_edit.reviewer == "paul.d"
    assert total_edit.action == "edit_field"
    assert total_edit.before_value == old_total
    assert total_edit.after_value == correct_total

    status_change = next(a for a in audits if a.field_path == "status")
    assert status_change.before_value == "needs_review"
    assert status_change.after_value == "approved"


def test_store_state_machine_guards_and_missing_doc() -> None:
    repo = DocumentRepository.from_url("sqlite:///:memory:")
    sample = generate_sample(seed=603, layout=LayoutStyle.MODERN_SIDEBAR, fault=FaultType.NONE)
    ingest_res, _ = ingest_bytes(
        sample.pdf_bytes,
        settings=Settings(target_dpi=100),
        persist=False,
    )
    rec = repo.create_from_ingest(ingest_res)

    # Mark failed from RECEIVED is valid
    failed = repo.mark_failed(rec.id, "LLM timeout")
    assert failed.status == DocumentStatus.FAILED.value
    assert failed.error_message == "LLM timeout"

    # Cannot transition directly from FAILED to APPROVED
    with pytest.raises(InvalidStateTransitionError):
        repo.apply_review(rec.id, reviewer="paul.d", action=DocumentStatus.APPROVED)

    with pytest.raises(DocumentNotFoundError):
        repo.get("nonexistent_id")
