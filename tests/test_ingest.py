"""Unit tests for document ingest: magic-byte MIME sniffing, hashing, limits, and splitting."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from docextract.config import Settings
from docextract.ingest import (
    CorruptDocumentError,
    IngestError,
    PageLimitExceededError,
    PayloadTooLargeError,
    SupportedMime,
    UnsupportedMediaTypeError,
    compute_sha256,
    ingest_bytes,
    sniff_mime,
    validate_upload_size,
)


def _make_pdf_bytes(num_pages: int = 2) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    for idx in range(1, num_pages + 1):
        c.drawString(72, 720, f"Northwind Labs Invoice - Page {idx} of {num_pages}")
        c.showPage()
    c.save()
    return buf.getvalue()


def _make_image_bytes(fmt: str = "PNG", size: tuple[int, int] = (400, 600)) -> bytes:
    img = Image.new("RGB", size, color=(245, 245, 245))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def test_sniff_mime_magic_bytes() -> None:
    pdf_bytes = _make_pdf_bytes(1)
    png_bytes = _make_image_bytes("PNG")
    jpeg_bytes = _make_image_bytes("JPEG")
    webp_bytes = _make_image_bytes("WEBP")
    tiff_bytes = _make_image_bytes("TIFF")

    assert sniff_mime(pdf_bytes) == SupportedMime.PDF
    assert sniff_mime(png_bytes) == SupportedMime.PNG
    assert sniff_mime(jpeg_bytes) == SupportedMime.JPEG
    assert sniff_mime(webp_bytes) == SupportedMime.WEBP
    assert sniff_mime(tiff_bytes) == SupportedMime.TIFF

    # Spoofed extension / plain text or executable bytes are rejected
    with pytest.raises(UnsupportedMediaTypeError):
        sniff_mime(b"MZ\x90\x00\x03\x00\x00\x00")

    with pytest.raises(UnsupportedMediaTypeError):
        sniff_mime(b"ab")


def test_compute_sha256_and_size_limits() -> None:
    payload = b"sample-invoice-payload"
    digest = compute_sha256(payload)
    assert len(digest) == 64

    with pytest.raises(IngestError, match="empty"):
        compute_sha256(b"")

    with pytest.raises(IngestError, match="empty"):
        validate_upload_size(b"", max_bytes=1024)

    with pytest.raises(PayloadTooLargeError, match="exceeds limit"):
        validate_upload_size(b"x" * 2048, max_bytes=1024)


def test_ingest_multipage_pdf_and_deduplication(tmp_path: Path) -> None:
    pdf_bytes = _make_pdf_bytes(num_pages=3)
    settings = Settings(target_dpi=144, max_pages=10, storage_dir=tmp_path)

    res1, pages1 = ingest_bytes(pdf_bytes, settings=settings, storage_dir=tmp_path)
    assert res1.mime_type == SupportedMime.PDF
    assert res1.page_count == 3
    assert len(pages1) == 3
    assert res1.is_duplicate is False
    assert res1.raw_path is not None and Path(res1.raw_path).exists()

    # 8.5 x 11 inches at 144 DPI -> 1224 x 1584 px
    assert res1.pages[0].width_px == 1224
    assert res1.pages[0].height_px == 1584
    assert res1.pages[0].image_path is not None and Path(res1.pages[0].image_path).exists()

    # Re-ingesting identical bytes flags `is_duplicate=True` with identical SHA-256
    res2, _ = ingest_bytes(pdf_bytes, settings=settings, storage_dir=tmp_path)
    assert res2.doc_sha256 == res1.doc_sha256
    assert res2.is_duplicate is True


def test_ingest_single_and_multiframe_images(tmp_path: Path) -> None:
    png_bytes = _make_image_bytes("PNG", size=(500, 700))
    settings = Settings(target_dpi=150, max_pages=5, storage_dir=tmp_path)

    res, pages = ingest_bytes(png_bytes, settings=settings, storage_dir=tmp_path)
    assert res.mime_type == SupportedMime.PNG
    assert res.page_count == 1
    assert pages[0].image.size == (500, 700)

    # Multi-frame TIFF
    f1 = Image.new("RGB", (300, 400), color=(255, 255, 255))
    f2 = Image.new("RGB", (300, 400), color=(200, 200, 200))
    tiff_buf = io.BytesIO()
    f1.save(tiff_buf, format="TIFF", save_all=True, append_images=[f2])
    res_tiff, pages_tiff = ingest_bytes(
        tiff_buf.getvalue(), settings=settings, storage_dir=tmp_path
    )
    assert res_tiff.mime_type == SupportedMime.TIFF
    assert res_tiff.page_count == 2
    assert len(pages_tiff) == 2


def test_ingest_page_limit_and_corrupt_files(tmp_path: Path) -> None:
    pdf_4_pages = _make_pdf_bytes(num_pages=4)
    strict_settings = Settings(target_dpi=72, max_pages=2, storage_dir=tmp_path)

    with pytest.raises(PageLimitExceededError, match="exceeding the maximum allowed"):
        ingest_bytes(pdf_4_pages, settings=strict_settings, persist=False)

    # Corrupt PDF with valid %PDF- magic header but invalid body
    corrupt_pdf = b"%PDF-1.7\ngarbage-not-a-real-xref-table"
    with pytest.raises(CorruptDocumentError):
        ingest_bytes(corrupt_pdf, settings=strict_settings, persist=False)

    # Corrupt PNG with valid 8-byte signature but truncated IHDR
    corrupt_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00"
    with pytest.raises(CorruptDocumentError):
        ingest_bytes(corrupt_png, settings=strict_settings, persist=False)
