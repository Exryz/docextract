"""Unit tests for field-level and document-level confidence scoring and review routing."""

from __future__ import annotations

from decimal import Decimal

from datagen import FaultType, LayoutStyle, generate_sample
from docextract.config import Settings
from docextract.extract import ground_extracted_document
from docextract.schemas import Invoice
from docextract.scoring import (
    compute_consistency_agreement,
    score_document_confidence,
)
from docextract.validate import validate_document
from docextract.vision import extract_pdf_text_layer


def test_grounded_valid_document_achieves_high_confidence_and_auto_approves() -> None:
    sample = generate_sample(seed=501, layout=LayoutStyle.CORPORATE_TABLE, fault=FaultType.NONE)
    layouts = extract_pdf_text_layer(sample.pdf_bytes)
    grounded = ground_extracted_document(sample.gold_document, layouts)
    val_report = validate_document(grounded)

    conf_report = score_document_confidence(
        grounded,
        layouts=layouts,
        validation=val_report,
        settings=Settings(min_field_confidence=0.75),
    )

    assert conf_report.needs_review is False
    assert conf_report.low_confidence_fields == []
    assert conf_report.document_confidence >= 0.90
    assert conf_report.fields["invoice_number"].grounding_score == 1.0
    assert conf_report.fields["total"].combined_score == 1.0


def test_validation_rule_failure_drops_field_confidence_and_triggers_review() -> None:
    sample = generate_sample(
        seed=502, layout=LayoutStyle.CORPORATE_TABLE, fault=FaultType.WRONG_TOTAL
    )
    layouts = extract_pdf_text_layer(sample.pdf_bytes)
    grounded = ground_extracted_document(sample.injected_document, layouts)
    val_report = validate_document(grounded)

    conf_report = score_document_confidence(
        grounded,
        layouts=layouts,
        validation=val_report,
    )

    assert conf_report.needs_review is True
    assert "total" in conf_report.low_confidence_fields
    assert conf_report.fields["total"].rule_score == 0.15
    assert conf_report.fields["total"].combined_score < 0.25


def test_self_consistency_disagreement_and_ungrounded_fields_lower_confidence() -> None:
    sample = generate_sample(seed=503, layout=LayoutStyle.MODERN_SIDEBAR, fault=FaultType.NONE)
    assert isinstance(sample.gold_document, Invoice)

    # Candidate 1 disagrees on `total` and `invoice_number`; Candidate 2 disagrees on `total`
    cand1 = sample.gold_document.model_copy(
        update={
            "invoice_number": "INV-HALLUCINATED",
            "total": sample.gold_document.total + Decimal("10.00"),
        }
    )
    cand2 = sample.gold_document.model_copy(
        update={"total": sample.gold_document.total + Decimal("20.00")}
    )

    agreement = compute_consistency_agreement(sample.gold_document, [cand1, cand2])
    # Primary + 2 candidates = 3 total:
    # - invoice_number matches in 2/3 (0.6667)
    # - total matches in 1/3 (0.3333)
    # - subtotal matches in 3/3 (1.0)
    assert agreement["invoice_number"] == 0.6667
    assert agreement["total"] == 0.3333
    assert agreement["subtotal"] == 1.0

    # Score without layouts (ungrounded -> grounding_score = 0.4) and with disagreement
    conf_report = score_document_confidence(
        sample.gold_document,
        layouts=[],
        candidates=[cand1, cand2],
        settings=Settings(min_field_confidence=0.75),
    )
    assert conf_report.needs_review is True
    assert "total" in conf_report.low_confidence_fields
    assert conf_report.fields["total"].grounding_score == 0.4
    assert conf_report.fields["total"].consistency_score == 0.3333
