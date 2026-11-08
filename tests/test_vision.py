"""Unit tests for `docextract.vision`: PDF text layer bboxes, deskew, and OCR fallback."""

from __future__ import annotations

import io

from PIL import Image, ImageDraw
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from docextract.schemas import BoundingBox, WordToken
from docextract.vision import (
    StaticOcrEngine,
    adaptive_binarize,
    deskew_image,
    estimate_skew_angle,
    extract_document_layout,
    extract_pdf_text_layer,
    preprocess_page_image,
)


def _make_invoice_pdf_bytes() -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, 720, "INVOICE INV-2026-0099")
    c.setFont("Helvetica", 11)
    c.drawString(72, 690, "Vendor: Northwind Cloud Services")
    c.drawString(72, 660, "Total Due: $1,480.00")
    c.showPage()
    c.save()
    return buf.getvalue()


def _make_blank_scanned_pdf_bytes() -> bytes:
    """Create a 1-page PDF containing only vector graphics and zero text characters."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    c.rect(72, 500, 400, 200, stroke=1, fill=0)
    c.showPage()
    c.save()
    return buf.getvalue()


def _make_skewed_text_image(skew_deg: float = 4.0) -> Image.Image:
    """Create a synthetic document image with dark horizontal lines rotated by `skew_deg`."""
    img = Image.new("RGB", (600, 800), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    for y in range(120, 680, 36):
        draw.rectangle((80, y, 520, y + 8), fill=(20, 20, 20))
    return img.rotate(-skew_deg, resample=Image.Resampling.BICUBIC, fillcolor=(255, 255, 255))


def test_extract_pdf_text_layer_and_bbox_grounding() -> None:
    pdf_bytes = _make_invoice_pdf_bytes()
    layouts = extract_pdf_text_layer(pdf_bytes)
    assert len(layouts) == 1

    page = layouts[0]
    assert page.page_number == 1
    assert page.source == "pdf_text"
    assert len(page.words) >= 8

    full_text = page.full_text
    assert "INVOICE INV-2026-0099" in full_text
    assert "Vendor: Northwind Cloud Services" in full_text
    assert "Total Due: $1,480.00" in full_text

    # Single-token and multi-word span bbox lookup
    inv_bbox = page.find_matching_bbox("INV-2026-0099")
    assert inv_bbox is not None
    assert inv_bbox.page == 1
    # Top of the page in normalized top-left coordinates (y0 < 0.2)
    assert 0.0 < inv_bbox.y0 < 0.2

    vendor_bbox = page.find_matching_bbox("Northwind Cloud Services")
    assert vendor_bbox is not None
    assert vendor_bbox.width > inv_bbox.width

    amount_bbox = page.find_matching_bbox("1,480.00")
    assert amount_bbox is not None
    assert amount_bbox.y0 > inv_bbox.y0


def test_skew_estimation_and_deskew() -> None:
    # Unskewed image should have ~0 deg skew and not be rotated
    straight = _make_skewed_text_image(skew_deg=0.0)
    assert abs(estimate_skew_angle(straight)) <= 0.4
    _, angle0, rotated0 = deskew_image(straight)
    assert rotated0 is False

    # Image skewed by +4.0 deg clockwise should be detected near +4.0 deg and corrected
    tilted = _make_skewed_text_image(skew_deg=4.0)
    detected = estimate_skew_angle(tilted)
    assert 2.8 <= detected <= 5.2

    corrected, applied_angle, was_deskewed = deskew_image(tilted)
    assert was_deskewed is True
    assert abs(applied_angle - 4.0) <= 1.2
    # After correction, residual skew is near zero
    residual = estimate_skew_angle(corrected)
    assert abs(residual) <= 0.8


def test_adaptive_binarize_and_preprocess_pipeline() -> None:
    tilted = _make_skewed_text_image(skew_deg=3.0)
    prep = preprocess_page_image(tilted, deskew=True, binarize=True)
    assert prep.deskewed is True
    assert prep.thresholded is True
    assert prep.image.mode == "RGB"

    bin_only = adaptive_binarize(tilted)
    assert bin_only.size == tilted.size


def test_hybrid_layout_uses_ocr_fallback_for_scanned_pdfs_and_images() -> None:
    fake_tokens = [
        WordToken(
            text="RCP-5501",
            bbox=BoundingBox(page=1, x0=0.1, y0=0.1, x1=0.3, y1=0.14),
            confidence=0.96,
        ),
        WordToken(
            text="$42.50",
            bbox=BoundingBox(page=1, x0=0.6, y0=0.8, x1=0.78, y1=0.84),
            confidence=0.94,
        ),
    ]
    ocr_double = StaticOcrEngine({1: fake_tokens})

    # 1. Digital PDF with native text does NOT call OCR
    digital_pdf = _make_invoice_pdf_bytes()
    digital_layouts = extract_document_layout(digital_pdf, ocr_engine=ocr_double)
    assert digital_layouts[0].source == "pdf_text"
    assert ocr_double.calls == []

    # 2. Scanned/image-only PDF (0 native words) falls back to OCR automatically
    scanned_pdf = _make_blank_scanned_pdf_bytes()
    scanned_layouts = extract_document_layout(
        scanned_pdf, ocr_engine=ocr_double, target_dpi=100, deskew_scans=False
    )
    assert scanned_layouts[0].source == "ocr"
    assert ocr_double.calls == [1]
    assert "RCP-5501" in scanned_layouts[0].full_text

    # 3. Direct PNG image upload runs preprocess + OCR
    img_buf = io.BytesIO()
    _make_skewed_text_image(skew_deg=0.0).save(img_buf, format="PNG")
    img_layouts = extract_document_layout(
        img_buf.getvalue(), ocr_engine=ocr_double, target_dpi=100
    )
    assert img_layouts[0].source == "ocr"
    assert len(ocr_double.calls) == 2
