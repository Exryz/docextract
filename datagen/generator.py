"""Synthetic invoice and receipt renderer with multiple layouts and controlled fault injection."""

from __future__ import annotations

import io
import json
import random
from datetime import date, timedelta
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any

from faker import Faker
from PIL import Image, ImageFilter
from pydantic import BaseModel, ConfigDict
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from docextract.ingest.splitter import render_pdf_pages
from docextract.schemas import (
    Currency,
    DocumentKind,
    Invoice,
    LineItem,
    PartyInfo,
    PaymentMethod,
    Receipt,
    TaxLine,
    quantize_amount,
)


class LayoutStyle(StrEnum):
    CORPORATE_TABLE = "corporate_table"
    MODERN_SIDEBAR = "modern_sidebar"
    THERMAL_RECEIPT = "thermal_receipt"


class FaultType(StrEnum):
    NONE = "none"
    WRONG_LINE_AMOUNT = "wrong_line_amount"
    WRONG_SUBTOTAL = "wrong_subtotal"
    WRONG_TAX = "wrong_tax"
    WRONG_TOTAL = "wrong_total"
    INVERTED_DATES = "inverted_dates"
    ROTATED_SCAN = "rotated_scan"


class GeneratedSample(BaseModel):
    """A rendered synthetic document paired with its gold-standard label and fault tag."""

    model_config = ConfigDict(frozen=True)

    sample_id: str
    layout: LayoutStyle
    fault: FaultType
    pdf_bytes: bytes
    scan_png_bytes: bytes | None = None
    gold_document: Invoice | Receipt
    injected_document: Invoice | Receipt

    def save(self, out_dir: Path) -> dict[str, str]:
        """Write `.pdf`, optional `.scan.png`, and `.gold.json` metadata to `out_dir`."""
        out_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = out_dir / f"{self.sample_id}.pdf"
        pdf_path.write_bytes(self.pdf_bytes)

        paths: dict[str, str] = {"pdf": pdf_path.as_posix()}
        if self.scan_png_bytes is not None:
            png_path = out_dir / f"{self.sample_id}.scan.png"
            png_path.write_bytes(self.scan_png_bytes)
            paths["scan_png"] = png_path.as_posix()

        meta_path = out_dir / f"{self.sample_id}.gold.json"
        payload: dict[str, Any] = {
            "sample_id": self.sample_id,
            "layout": self.layout.value,
            "fault": self.fault.value,
            "gold_document": json.loads(self.gold_document.model_dump_json()),
            "injected_document": json.loads(self.injected_document.model_dump_json()),
        }
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        paths["gold_json"] = meta_path.as_posix()
        return paths


_CATALOG: list[tuple[str, str, Decimal]] = [
    ("SKU-GPU-A100", "Dedicated Cloud GPU Node (A100 80GB)", Decimal("320.00")),
    ("SKU-NVME-2TB", "Enterprise Block Storage 2TB NVMe", Decimal("95.00")),
    ("SKU-K8S-CP", "Managed Kubernetes Control Plane", Decimal("140.00")),
    ("SKU-WAF-PRO", "Edge DDoS & WAF Protection Tier", Decimal("75.00")),
    ("SKU-OBS-LOG", "Log Ingestion & Metrics Retention (1TB)", Decimal("60.00")),
    ("SKU-SUP-247", "24/7 Production SLA Support Retainer", Decimal("250.00")),
]

_CAFE_ITEMS: list[tuple[str, Decimal]] = [
    ("Pour Over Barako Coffee", Decimal("4.50")),
    ("Cold Brew Oat Latte", Decimal("5.25")),
    ("Almond Croissant", Decimal("4.75")),
    ("Sourdough Avocado Toast", Decimal("9.50")),
    ("Espresso Double Shot", Decimal("3.50")),
]


def _build_gold_invoice(seed: int, currency: Currency = Currency.USD) -> Invoice:
    rng = random.Random(seed)
    fake = Faker()
    Faker.seed(seed)

    issue = date(2026, rng.randint(1, 10), rng.randint(1, 25))
    due = issue + timedelta(days=30)

    picked = rng.sample(_CATALOG, k=rng.randint(2, 4))
    items: list[LineItem] = []
    for sku, desc, price in picked:
        qty = Decimal(str(rng.randint(1, 5)))
        amt = quantize_amount(qty * price, currency)
        items.append(
            LineItem(
                sku=sku,
                description=desc,
                quantity=qty,
                unit_price=price,
                amount=amt,
            )
        )

    subtotal = sum((i.amount for i in items), Decimal("0.00"))
    tax_rate = Decimal("0.12")
    tax_amt = quantize_amount(subtotal * tax_rate, currency)
    total = quantize_amount(subtotal + tax_amt, currency)

    vendor = PartyInfo(
        name=fake.company(),
        tax_id=f"US-{rng.randint(10, 99)}-{rng.randint(1000000, 9999999)}",
        iban="DE89370400440532013000",
        address=fake.address().replace("\n", ", "),
        email=fake.company_email(),
    )
    customer = PartyInfo(
        name=fake.company(),
        address=fake.address().replace("\n", ", "),
    )

    return Invoice(
        kind=DocumentKind.INVOICE,
        invoice_number=f"INV-2026-{seed:04d}",
        issue_date=issue,
        due_date=due,
        purchase_order=f"PO-{rng.randint(10000, 99999)}",
        currency=currency,
        vendor=vendor,
        customer=customer,
        line_items=items,
        taxes=[
            TaxLine(
                label="VAT (12%)",
                rate=tax_rate,
                taxable_base=subtotal,
                amount=tax_amt,
            )
        ],
        subtotal=subtotal,
        tax_total=tax_amt,
        total=total,
        payment_method=PaymentMethod.BANK_TRANSFER,
    )


def _build_gold_receipt(seed: int, currency: Currency = Currency.USD) -> Receipt:
    rng = random.Random(seed)
    fake = Faker()
    Faker.seed(seed)

    tx_date = date(2026, rng.randint(1, 10), rng.randint(1, 28))
    picked = rng.sample(_CAFE_ITEMS, k=rng.randint(2, 4))
    items: list[LineItem] = []
    for desc, price in picked:
        qty = Decimal(str(rng.randint(1, 3)))
        amt = quantize_amount(qty * price, currency)
        items.append(
            LineItem(
                description=desc,
                quantity=qty,
                unit_price=price,
                amount=amt,
            )
        )

    subtotal = sum((i.amount for i in items), Decimal("0.00"))
    tax_rate = Decimal("0.08")
    tax_amt = quantize_amount(subtotal * tax_rate, currency)
    tip_amt = Decimal("3.00")
    total = quantize_amount(subtotal + tax_amt + tip_amt, currency)

    return Receipt(
        kind=DocumentKind.RECEIPT,
        receipt_number=f"RCP-{seed:04d}",
        transaction_date=tx_date,
        currency=currency,
        merchant=PartyInfo(
            name=f"{fake.last_name()} Artisan Roasters",
            tax_id=f"US-{rng.randint(10, 99)}-{rng.randint(1000000, 9999999)}",
            address=fake.street_address(),
        ),
        line_items=items,
        taxes=[TaxLine(label="Sales Tax 8%", rate=tax_rate, taxable_base=subtotal, amount=tax_amt)],
        subtotal=subtotal,
        tax_total=tax_amt,
        tip_total=tip_amt,
        total=total,
        payment_method=PaymentMethod.CARD,
        card_last4=f"{rng.randint(1000, 9999)}",
    )


def inject_fault(doc: Invoice | Receipt, fault: FaultType) -> Invoice | Receipt:
    """Return a copy of `doc` with a specific mathematical or business-rule error injected."""
    if fault in (FaultType.NONE, FaultType.ROTATED_SCAN):
        return doc

    if isinstance(doc, Invoice):
        if fault == FaultType.WRONG_LINE_AMOUNT:
            bad_first = doc.line_items[0].model_copy(
                update={"amount": doc.line_items[0].amount + Decimal("25.00")}
            )
            return doc.model_copy(update={"line_items": [bad_first, *doc.line_items[1:]]})
        if fault == FaultType.WRONG_SUBTOTAL:
            return doc.model_copy(update={"subtotal": doc.subtotal + Decimal("50.00")})
        if fault == FaultType.WRONG_TAX:
            return doc.model_copy(update={"tax_total": doc.tax_total + Decimal("19.50")})
        if fault == FaultType.WRONG_TOTAL:
            return doc.model_copy(update={"total": doc.total + Decimal("100.00")})
        if fault == FaultType.INVERTED_DATES:
            bad_due = doc.issue_date - timedelta(days=15)
            return doc.model_copy(update={"due_date": bad_due})

    if isinstance(doc, Receipt):
        if fault == FaultType.WRONG_LINE_AMOUNT:
            bad_first = doc.line_items[0].model_copy(
                update={"amount": doc.line_items[0].amount + Decimal("5.00")}
            )
            return doc.model_copy(update={"line_items": [bad_first, *doc.line_items[1:]]})
        if fault == FaultType.WRONG_SUBTOTAL:
            return doc.model_copy(update={"subtotal": doc.subtotal + Decimal("10.00")})
        if fault == FaultType.WRONG_TAX:
            return doc.model_copy(update={"tax_total": doc.tax_total + Decimal("4.25")})
        if fault == FaultType.WRONG_TOTAL:
            return doc.model_copy(update={"total": doc.total + Decimal("15.00")})

    return doc


def render_invoice_pdf(doc: Invoice, style: LayoutStyle = LayoutStyle.CORPORATE_TABLE) -> bytes:
    """Render an `Invoice` to PDF bytes using the requested layout template."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter

    if style == LayoutStyle.MODERN_SIDEBAR:
        # Left accent bar
        c.setFillColor(colors.HexColor("#1E293B"))
        c.rect(0, 0, 150, height, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(20, height - 60, "INVOICE")
        c.setFont("Helvetica", 10)
        c.drawString(20, height - 85, f"No: {doc.invoice_number}")
        c.drawString(20, height - 102, f"Issued: {doc.issue_date.isoformat()}")
        if doc.due_date:
            c.drawString(20, height - 119, f"Due: {doc.due_date.isoformat()}")
        if doc.purchase_order:
            c.drawString(20, height - 136, f"PO: {doc.purchase_order}")
        x_left = 175
    else:
        # Corporate header
        c.setFillColor(colors.HexColor("#0F172A"))
        c.setFont("Helvetica-Bold", 18)
        c.drawString(54, height - 56, "INVOICE")
        c.setFont("Helvetica", 10)
        c.drawString(54, height - 74, f"Invoice Number: {doc.invoice_number}")
        c.drawString(54, height - 89, f"Issue Date: {doc.issue_date.isoformat()}")
        if doc.due_date:
            c.drawString(54, height - 104, f"Due Date: {doc.due_date.isoformat()}")
        if doc.purchase_order:
            c.drawString(54, height - 119, f"PO Number: {doc.purchase_order}")
        x_left = 54

    # Vendor & Customer blocks
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(x_left, height - 150, f"Vendor: {doc.vendor.name}")
    c.setFont("Helvetica", 9)
    if doc.vendor.tax_id:
        c.drawString(x_left, height - 165, f"Tax ID: {doc.vendor.tax_id}")
    if doc.vendor.iban:
        c.drawString(x_left, height - 178, f"IBAN: {doc.vendor.iban}")

    if doc.customer:
        c.setFont("Helvetica-Bold", 11)
        c.drawString(x_left, height - 205, f"Bill To: {doc.customer.name}")

    # Table header
    y = height - 245
    c.setFont("Helvetica-Bold", 10)
    c.drawString(x_left, y, "Description")
    c.drawString(width - 210, y, "Qty")
    c.drawString(width - 155, y, "Unit Price")
    c.drawString(width - 85, y, "Amount")
    c.line(x_left, y - 4, width - 36, y - 4)

    y -= 22
    c.setFont("Helvetica", 10)
    for item in doc.line_items:
        c.drawString(x_left, y, item.description[:42])
        c.drawString(width - 210, y, str(item.quantity))
        c.drawString(width - 155, y, f"{doc.currency.value} {item.unit_price}")
        c.drawString(width - 85, y, f"{doc.currency.value} {item.amount}")
        y -= 18

    y -= 12
    c.line(width - 220, y + 10, width - 36, y + 10)
    c.setFont("Helvetica", 10)
    c.drawString(width - 210, y - 4, f"Subtotal: {doc.currency.value} {doc.subtotal}")
    c.drawString(width - 210, y - 20, f"Tax Total: {doc.currency.value} {doc.tax_total}")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(width - 210, y - 42, f"Total: {doc.currency.value} {doc.total}")

    c.showPage()
    c.save()
    return buf.getvalue()


def render_receipt_pdf(doc: Receipt) -> bytes:
    """Render a thermal-style merchant receipt to PDF bytes."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    _, height = letter
    x_left = 140
    y = height - 70

    c.setFont("Courier-Bold", 14)
    c.drawString(x_left, y, doc.merchant.name)
    y -= 18
    c.setFont("Courier", 10)
    if doc.receipt_number:
        c.drawString(x_left, y, f"Receipt: {doc.receipt_number}")
        y -= 15
    c.drawString(x_left, y, f"Date: {doc.transaction_date.isoformat()}")
    y -= 22

    c.drawString(x_left, y, "-" * 42)
    y -= 16
    for item in doc.line_items:
        c.drawString(
            x_left,
            y,
            f"{item.quantity}x {item.description[:22]:<22} {doc.currency.value} {item.amount}",
        )
        y -= 15

    c.drawString(x_left, y, "-" * 42)
    y -= 16
    c.drawString(x_left, y, f"Subtotal: {doc.currency.value} {doc.subtotal}")
    y -= 15
    c.drawString(x_left, y, f"Tax: {doc.currency.value} {doc.tax_total}")
    y -= 15
    c.drawString(x_left, y, f"Tip: {doc.currency.value} {doc.tip_total}")
    y -= 18
    c.setFont("Courier-Bold", 12)
    c.drawString(x_left, y, f"Total: {doc.currency.value} {doc.total}")
    if doc.card_last4:
        y -= 16
        c.setFont("Courier", 9)
        c.drawString(x_left, y, f"Paid via Card *{doc.card_last4}")

    c.showPage()
    c.save()
    return buf.getvalue()


def _simulate_rotated_scan_png(pdf_bytes: bytes, angle_deg: float = 3.5) -> bytes:
    rendered = render_pdf_pages(pdf_bytes, target_dpi=150, max_pages=1)[0]
    rotated = rendered.image.rotate(
        -angle_deg,
        resample=Image.Resampling.BICUBIC,
        fillcolor=(255, 255, 255),
    )
    slightly_blurred = rotated.filter(ImageFilter.GaussianBlur(radius=0.35))
    out = io.BytesIO()
    slightly_blurred.save(out, format="PNG", dpi=(150, 150))
    return out.getvalue()


def generate_sample(
    seed: int = 1,
    layout: LayoutStyle = LayoutStyle.CORPORATE_TABLE,
    fault: FaultType = FaultType.NONE,
    currency: Currency = Currency.USD,
) -> GeneratedSample:
    """Generate a deterministic synthetic invoice or receipt with optional fault injection."""
    if layout == LayoutStyle.THERMAL_RECEIPT:
        gold_doc: Invoice | Receipt = _build_gold_receipt(seed=seed, currency=currency)
        injected_doc = inject_fault(gold_doc, fault)
        assert isinstance(injected_doc, Receipt)
        pdf_bytes = render_receipt_pdf(injected_doc)
    else:
        gold_doc = _build_gold_invoice(seed=seed, currency=currency)
        injected_doc = inject_fault(gold_doc, fault)
        assert isinstance(injected_doc, Invoice)
        pdf_bytes = render_invoice_pdf(injected_doc, style=layout)

    scan_png = (
        _simulate_rotated_scan_png(pdf_bytes, angle_deg=3.5)
        if fault == FaultType.ROTATED_SCAN
        else None
    )

    return GeneratedSample(
        sample_id=f"sample_{seed:04d}_{layout.value}_{fault.value}",
        layout=layout,
        fault=fault,
        pdf_bytes=pdf_bytes,
        scan_png_bytes=scan_png,
        gold_document=gold_doc,
        injected_document=injected_doc,
    )


def generate_dataset(
    count: int = 12,
    out_dir: Path | None = None,
    base_seed: int = 100,
) -> list[GeneratedSample]:
    """Generate a balanced dataset cycling across layouts and fault types."""
    layouts = [
        LayoutStyle.CORPORATE_TABLE,
        LayoutStyle.MODERN_SIDEBAR,
        LayoutStyle.THERMAL_RECEIPT,
    ]
    faults = [
        FaultType.NONE,
        FaultType.NONE,
        FaultType.WRONG_LINE_AMOUNT,
        FaultType.WRONG_SUBTOTAL,
        FaultType.WRONG_TAX,
        FaultType.WRONG_TOTAL,
        FaultType.INVERTED_DATES,
        FaultType.ROTATED_SCAN,
    ]

    samples: list[GeneratedSample] = []
    for i in range(count):
        layout = layouts[i % len(layouts)]
        fault = faults[i % len(faults)]
        if layout == LayoutStyle.THERMAL_RECEIPT and fault == FaultType.INVERTED_DATES:
            fault = FaultType.WRONG_TOTAL
        sample = generate_sample(seed=base_seed + i, layout=layout, fault=fault)
        if out_dir is not None:
            sample.save(out_dir)
        samples.append(sample)
    return samples
