"""Unit tests for core schemas: Money, BoundingBox, LineItem, Invoice, Receipt, and Settings."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from docextract.config import Settings, get_settings
from docextract.schemas import (
    BoundingBox,
    Currency,
    GroundedValue,
    Invoice,
    LineItem,
    Money,
    PartyInfo,
    PaymentMethod,
    Receipt,
    TaxLine,
    WordToken,
    quantize_amount,
)


def test_settings_defaults() -> None:
    s = get_settings()
    assert isinstance(s, Settings)
    assert s.target_dpi == 300
    assert s.max_pages == 25
    assert s.rounding_tolerance == Decimal("0.02")


def test_money_decimal_arithmetic_and_currency_safety() -> None:
    m1 = Money(amount="125.505", currency="usd").quantized()
    m2 = Money(amount=Decimal("24.50"), currency=Currency.USD)

    # Half-up rounding: 125.505 -> 125.51
    assert m1.amount == Decimal("125.51")
    assert m1.format() == "USD 125.51"

    total = m1 + m2
    assert total.amount == Decimal("150.01")

    diff = total - m2
    assert diff.amount == Decimal("125.51")

    scaled = m2.multiply(3)
    assert scaled.amount == Decimal("73.50")

    assert total.within_tolerance(Money(amount="150.02", currency=Currency.USD))
    assert not total.within_tolerance(Money(amount="150.05", currency=Currency.USD))

    # Zero-decimal currency (JPY)
    jpy = Money(amount="1500.6", currency=Currency.JPY).quantized()
    assert jpy.amount == Decimal("1501")
    assert quantize_amount("99.5", Currency.JPY) == Decimal("100")

    # Currency mismatch raises ValueError
    eur = Money(amount="10.00", currency=Currency.EUR)
    with pytest.raises(ValueError, match="Currency mismatch"):
        _ = m1 + eur


def test_bounding_box_geometry_and_merging() -> None:
    b1 = BoundingBox(page=1, x0=0.1, y0=0.2, x1=0.5, y1=0.6)
    b2 = BoundingBox(page=1, x0=0.3, y0=0.4, x1=0.7, y1=0.8)
    b_other_page = BoundingBox(page=2, x0=0.1, y0=0.2, x1=0.5, y1=0.6)

    assert b1.width == 0.4
    assert b1.height == 0.4
    assert b1.area == 0.16
    assert b1.contains_point(0.25, 0.35)
    assert not b1.contains_point(0.8, 0.8)

    # Intersection is (0.3..0.5, 0.4..0.6) -> 0.2 * 0.2 = 0.04
    # Union is 0.16 + 0.16 - 0.04 = 0.28 -> IoU = 0.04 / 0.28 = 0.142857
    assert b1.iou(b2) == pytest.approx(0.142857, rel=1e-4)
    assert b1.iou(b_other_page) == 0.0

    merged = BoundingBox.merge([b1, b2])
    assert merged.x0 == 0.1
    assert merged.y0 == 0.2
    assert merged.x1 == 0.7
    assert merged.y1 == 0.8

    # Pixel conversion round-trip
    px = b1.to_pixel_rect(1000, 2000)
    assert px == (100, 400, 500, 1200)
    from_px = BoundingBox.from_pixel_rect(1, *px, page_width=1000, page_height=2000)
    assert from_px == b1

    # Invalid inverted coordinates raise ValidationError
    with pytest.raises(ValidationError):
        BoundingBox(page=1, x0=0.8, y0=0.2, x1=0.3, y1=0.5)


def test_word_token_and_grounded_value() -> None:
    box = BoundingBox(page=1, x0=0.12, y0=0.08, x1=0.28, y1=0.11)
    token = WordToken(text="INV-2026-0042", bbox=box, confidence=0.98)
    assert token.text == "INV-2026-0042"

    gv = GroundedValue[Decimal](
        value=Decimal("1240.00"),
        source_bbox=box,
        raw_text="$1,240.00",
        confidence=0.95,
    )
    assert gv.value == Decimal("1240.00")
    assert gv.source_bbox == box


def test_invoice_and_receipt_computed_totals() -> None:
    items = [
        LineItem(
            description="Cloud GPU Instance (A100 80GB)",
            quantity=Decimal("10"),
            unit_price=Decimal("45.00"),
            amount=Decimal("450.00"),
        ),
        LineItem(
            description="Block Storage 2TB NVMe",
            quantity=Decimal("2"),
            unit_price=Decimal("80.00"),
            discount=Decimal("10.00"),
            amount=Decimal("150.00"),
        ),
    ]
    assert items[0].expected_amount() == Decimal("450.00")
    assert items[1].expected_amount() == Decimal("150.00")

    tax = TaxLine(label="VAT 12%", rate=Decimal("0.12"), amount=Decimal("72.00"))
    assert tax.expected_amount(Decimal("600.00")) == Decimal("72.00")

    inv = Invoice(
        invoice_number="INV-2026-101",
        issue_date=date(2026, 11, 1),
        due_date=date(2026, 12, 1),
        currency=Currency.USD,
        vendor=PartyInfo(name="Northwind Cloud Services Ltd.", tax_id="US-98-7654321"),
        customer=PartyInfo(name="Acme Robotics Inc."),
        line_items=items,
        taxes=[tax],
        subtotal=Decimal("600.00"),
        tax_total=Decimal("72.00"),
        total=Decimal("672.00"),
    )
    assert inv.computed_subtotal == Decimal("600.00")
    assert inv.computed_tax_total == Decimal("72.00")
    assert inv.computed_total == Decimal("672.00")

    rec = Receipt(
        receipt_number="RCP-8831",
        transaction_date=date(2026, 11, 5),
        currency=Currency.PHP,
        merchant=PartyInfo(name="Bauan Coffee Roasters"),
        line_items=[
            LineItem(
                description="Pour Over Barako",
                quantity=Decimal("2"),
                unit_price=Decimal("160.00"),
                amount=Decimal("320.00"),
            )
        ],
        subtotal=Decimal("320.00"),
        tax_total=Decimal("38.40"),
        tip_total=Decimal("30.00"),
        total=Decimal("388.40"),
        payment_method=PaymentMethod.CARD,
        card_last4="4242",
    )
    assert rec.computed_subtotal == Decimal("320.00")
    assert rec.computed_total == Decimal("388.40")
