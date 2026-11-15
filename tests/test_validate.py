"""Unit tests for the deterministic validation rule engine, math rules, and IBAN/Tax checksums."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from datagen import FaultType, LayoutStyle, generate_sample
from docextract.schemas import Invoice, TaxLine
from docextract.validate import (
    Severity,
    validate_document,
    validate_iban,
    validate_tax_id,
)


def test_iban_and_tax_id_validators() -> None:
    # Valid German and UK IBANs (ISO 13616 mod-97 == 1)
    assert validate_iban("DE89 3704 0044 0532 0130 00") == (True, None)
    assert validate_iban("GB82 WEST 1234 5698 7654 32") == (True, None)

    # Corrupted check digits fail mod-97
    ok_bad_crc, reason_crc = validate_iban("DE00 3704 0044 0532 0130 00")
    assert ok_bad_crc is False
    assert reason_crc is not None and "mod-97" in reason_crc

    # Wrong country length fails
    ok_len, reason_len = validate_iban("DE893704004405320130")
    assert ok_len is False
    assert reason_len is not None and "22 characters" in reason_len

    # Tax ID formats: US EIN, PH TIN, EU VAT
    assert validate_tax_id("US-12-3456789") == (True, None)
    assert validate_tax_id("123-456-789-000") == (True, None)
    assert validate_tax_id("DE123456789") == (True, None)
    assert validate_tax_id("INVALID_TAX!")[0] is False


def test_clean_invoices_and_receipts_pass_all_rules() -> None:
    for layout in (
        LayoutStyle.CORPORATE_TABLE,
        LayoutStyle.MODERN_SIDEBAR,
        LayoutStyle.THERMAL_RECEIPT,
    ):
        sample = generate_sample(seed=50, layout=layout, fault=FaultType.NONE)
        report = validate_document(sample.gold_document)
        assert report.passed is True
        assert report.error_count == 0
        assert report.warning_count == 0
        assert report.failed_errors == []


@pytest.mark.parametrize(
    ("fault", "expected_rule_id"),
    [
        (FaultType.WRONG_LINE_AMOUNT, "line_item_math"),
        (FaultType.WRONG_SUBTOTAL, "subtotal_math"),
        (FaultType.WRONG_TAX, "tax_total_math"),
        (FaultType.WRONG_TOTAL, "grand_total_math"),
        (FaultType.INVERTED_DATES, "invoice_due_date_order"),
    ],
)
def test_validation_catches_every_injected_fault(
    fault: FaultType,
    expected_rule_id: str,
) -> None:
    sample = generate_sample(seed=77, layout=LayoutStyle.CORPORATE_TABLE, fault=fault)
    report = validate_document(sample.injected_document)
    assert report.passed is False
    assert report.error_count >= 1
    assert expected_rule_id in report.failed_rule_ids


def test_validation_catches_bad_iban_and_implausible_tax_rate() -> None:
    sample = generate_sample(seed=88, layout=LayoutStyle.CORPORATE_TABLE, fault=FaultType.NONE)
    assert isinstance(sample.gold_document, Invoice)

    bad_vendor = sample.gold_document.vendor.model_copy(
        update={"iban": "DE99370400440532013000", "tax_id": "BAD_TIN"}
    )
    high_tax = TaxLine(
        label="Excessive Tax",
        rate=Decimal("0.50"),
        taxable_base=sample.gold_document.subtotal,
        amount=sample.gold_document.subtotal * Decimal("0.50"),
    )
    mutated = sample.gold_document.model_copy(
        update={
            "vendor": bad_vendor,
            "issue_date": date(1995, 5, 1),
            "taxes": [high_tax],
            "tax_total": high_tax.amount,
            "total": sample.gold_document.subtotal + high_tax.amount,
        }
    )

    report = validate_document(mutated)
    assert report.passed is False
    assert "iban_checksum" in report.failed_rule_ids
    assert "tax_id_format" in report.failed_rule_ids
    assert "tax_rate_plausibility" in report.failed_rule_ids
    assert "date_range_sanity" in report.failed_rule_ids
    assert any(w.severity == Severity.WARNING for w in report.failed_warnings)
