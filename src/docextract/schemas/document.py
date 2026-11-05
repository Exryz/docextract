"""Strict Pydantic schemas for extracted invoices, receipts, line items, and tax breakdowns."""

from __future__ import annotations

from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from docextract.schemas.grounding import BoundingBox
from docextract.schemas.money import Currency, quantize_amount


class DocumentKind(StrEnum):
    INVOICE = "invoice"
    RECEIPT = "receipt"


class PaymentMethod(StrEnum):
    BANK_TRANSFER = "bank_transfer"
    CARD = "card"
    CASH = "cash"
    CHECK = "check"
    EWALLET = "ewallet"
    OTHER = "other"


class PartyInfo(BaseModel):
    """Vendor or customer legal entity details on an invoice or receipt."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1, description="Registered business or customer name")
    tax_id: str | None = Field(default=None, description="VAT, TIN, or EIN registration number")
    iban: str | None = Field(default=None, description="Bank IBAN for wire transfers")
    address: str | None = None
    email: str | None = None
    phone: str | None = None
    source_bbox: BoundingBox | None = None


class LineItem(BaseModel):
    """Single billed line item on an invoice or receipt."""

    model_config = ConfigDict(frozen=True)

    description: str = Field(min_length=1)
    quantity: Decimal = Field(default=Decimal("1"), gt=Decimal("0"))
    unit_price: Decimal = Field(ge=Decimal("0"))
    discount: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    tax_rate: Decimal | None = Field(
        default=None,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Fractional tax rate when specified per line, e.g. 0.12 for 12%",
    )
    amount: Decimal = Field(
        ge=Decimal("0"), description="Extended line total before document-level tax"
    )
    sku: str | None = None
    source_bbox: BoundingBox | None = None

    @field_validator("quantity", "unit_price", "discount", "amount", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: object) -> Decimal:
        if isinstance(v, float):
            v = str(v)
        return Decimal(str(v))

    @field_validator("tax_rate", mode="before")
    @classmethod
    def _coerce_optional_decimal(cls, v: object) -> Decimal | None:
        if v is None:
            return None
        if isinstance(v, float):
            v = str(v)
        return Decimal(str(v))

    def expected_amount(self, currency: Currency = Currency.USD) -> Decimal:
        """Compute `(quantity * unit_price) - discount` rounded to currency precision."""
        raw = (self.quantity * self.unit_price) - self.discount
        return quantize_amount(raw, currency)


class TaxLine(BaseModel):
    """Tax breakdown row (e.g. VAT 12%, Sales Tax 8.25%)."""

    model_config = ConfigDict(frozen=True)

    label: str = Field(default="Tax", min_length=1)
    rate: Decimal | None = Field(
        default=None,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Fractional tax rate, e.g. 0.12 for 12%",
    )
    taxable_base: Decimal | None = Field(default=None, ge=Decimal("0"))
    amount: Decimal = Field(ge=Decimal("0"))
    source_bbox: BoundingBox | None = None

    @field_validator("amount", mode="before")
    @classmethod
    def _coerce_amount(cls, v: object) -> Decimal:
        if isinstance(v, float):
            v = str(v)
        return Decimal(str(v))

    @field_validator("rate", "taxable_base", mode="before")
    @classmethod
    def _coerce_optional(cls, v: object) -> Decimal | None:
        if v is None:
            return None
        if isinstance(v, float):
            v = str(v)
        return Decimal(str(v))

    def expected_amount(
        self, fallback_base: Decimal, currency: Currency = Currency.USD
    ) -> Decimal | None:
        """Compute expected tax amount from `rate * base` if `rate` is present."""
        if self.rate is None:
            return None
        base = self.taxable_base if self.taxable_base is not None else fallback_base
        raw = (base * self.rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        return quantize_amount(raw, currency)


class Invoice(BaseModel):
    """Structured invoice document with vendor, customer, line items, taxes, and groundings."""

    model_config = ConfigDict(frozen=True)

    kind: DocumentKind = DocumentKind.INVOICE
    invoice_number: str = Field(min_length=1)
    issue_date: date
    due_date: date | None = None
    purchase_order: str | None = None
    currency: Currency = Currency.USD

    vendor: PartyInfo
    customer: PartyInfo | None = None

    line_items: list[LineItem] = Field(min_length=1)
    taxes: list[TaxLine] = Field(default_factory=list)

    subtotal: Decimal = Field(ge=Decimal("0"))
    discount_total: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    tax_total: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    shipping_total: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    total: Decimal = Field(ge=Decimal("0"))

    payment_method: PaymentMethod | None = None
    notes: str | None = None
    field_bboxes: dict[str, BoundingBox] = Field(
        default_factory=dict,
        description="Map of field path (e.g. 'invoice_number', 'total') to page BoundingBox",
    )

    @field_validator(
        "subtotal", "discount_total", "tax_total", "shipping_total", "total", mode="before"
    )
    @classmethod
    def _coerce_totals(cls, v: object) -> Decimal:
        if isinstance(v, float):
            v = str(v)
        return Decimal(str(v))

    @property
    def computed_subtotal(self) -> Decimal:
        raw = sum((item.amount for item in self.line_items), Decimal("0"))
        return quantize_amount(raw, self.currency)

    @property
    def computed_tax_total(self) -> Decimal:
        if not self.taxes:
            return quantize_amount(self.tax_total, self.currency)
        raw = sum((t.amount for t in self.taxes), Decimal("0"))
        return quantize_amount(raw, self.currency)

    @property
    def computed_total(self) -> Decimal:
        raw = self.subtotal - self.discount_total + self.tax_total + self.shipping_total
        return quantize_amount(raw, self.currency)


class Receipt(BaseModel):
    """Point-of-sale or merchant receipt schema."""

    model_config = ConfigDict(frozen=True)

    kind: DocumentKind = DocumentKind.RECEIPT
    receipt_number: str | None = None
    transaction_date: date
    currency: Currency = Currency.USD

    merchant: PartyInfo
    line_items: list[LineItem] = Field(min_length=1)
    taxes: list[TaxLine] = Field(default_factory=list)

    subtotal: Decimal = Field(ge=Decimal("0"))
    discount_total: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    tax_total: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    tip_total: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    total: Decimal = Field(ge=Decimal("0"))

    payment_method: PaymentMethod = PaymentMethod.CARD
    card_last4: str | None = Field(default=None, pattern=r"^\d{4}$")
    field_bboxes: dict[str, BoundingBox] = Field(default_factory=dict)

    @field_validator("subtotal", "discount_total", "tax_total", "tip_total", "total", mode="before")
    @classmethod
    def _coerce_receipt_totals(cls, v: object) -> Decimal:
        if isinstance(v, float):
            v = str(v)
        return Decimal(str(v))

    @property
    def computed_subtotal(self) -> Decimal:
        raw = sum((item.amount for item in self.line_items), Decimal("0"))
        return quantize_amount(raw, self.currency)

    @property
    def computed_total(self) -> Decimal:
        raw = self.subtotal - self.discount_total + self.tax_total + self.tip_total
        return quantize_amount(raw, self.currency)


ExtractedDocument = Invoice | Receipt
