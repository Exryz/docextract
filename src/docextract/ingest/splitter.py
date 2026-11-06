"""Multi-page PDF and image splitter, rasterizer, and content-addressed page store."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageSequence
from pydantic import BaseModel, ConfigDict, Field

from docextract.config import Settings, get_settings
from docextract.ingest.sniffer import (
    CorruptDocumentError,
    PageLimitExceededError,
    SupportedMime,
    compute_sha256,
    sniff_mime,
    validate_upload_size,
)

PDF_POINTS_PER_INCH = 72.0


class PageAsset(BaseModel):
    """Metadata and rendered raster properties for a single ingested document page."""

    model_config = ConfigDict(frozen=True)

    page_number: int = Field(ge=1, description="1-based page index")
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    dpi: int = Field(gt=0)
    page_sha256: str = Field(min_length=64, max_length=64)
    image_path: str | None = Field(
        default=None,
        description="Relative or absolute path to persisted PNG raster if stored on disk",
    )


class IngestResult(BaseModel):
    """Result of ingesting, validating, and splitting a PDF or image upload."""

    model_config = ConfigDict(frozen=True)

    doc_sha256: str = Field(min_length=64, max_length=64)
    mime_type: SupportedMime
    size_bytes: int = Field(gt=0)
    page_count: int = Field(ge=1)
    pages: list[PageAsset] = Field(min_length=1)
    is_duplicate: bool = False
    raw_path: str | None = None


@dataclass(frozen=True)
class RenderedPage:
    """In-memory PIL image + PNG bytes for a single page."""

    page_number: int
    image: Image.Image
    png_bytes: bytes
    dpi: int
    page_sha256: str


def _encode_png(img: Image.Image, dpi: int) -> bytes:
    rgb = img.convert("RGB") if img.mode not in ("RGB", "L") else img
    buf = io.BytesIO()
    rgb.save(buf, format="PNG", dpi=(dpi, dpi))
    return buf.getvalue()


def render_pdf_pages(
    data: bytes,
    target_dpi: int = 300,
    max_pages: int = 25,
) -> list[RenderedPage]:
    """Rasterize each page of a PDF to PNG at `target_dpi` using `pypdfium2`."""
    try:
        pdf = pdfium.PdfDocument(data)
    except Exception as exc:
        raise CorruptDocumentError(f"Failed to open PDF document: {exc}") from exc

    try:
        n_pages = len(pdf)
        if n_pages == 0:
            raise CorruptDocumentError("PDF contains 0 pages")
        if n_pages > max_pages:
            raise PageLimitExceededError(
                f"PDF has {n_pages} pages, exceeding the maximum allowed ({max_pages})"
            )

        scale = target_dpi / PDF_POINTS_PER_INCH
        rendered: list[RenderedPage] = []
        for idx in range(n_pages):
            page = pdf[idx]
            try:
                bitmap = page.render(scale=scale)
                pil_img: Image.Image = bitmap.to_pil().convert("RGB")
            except Exception as exc:
                raise CorruptDocumentError(f"Failed rendering PDF page {idx + 1}: {exc}") from exc
            finally:
                page.close()

            png_bytes = _encode_png(pil_img, target_dpi)
            rendered.append(
                RenderedPage(
                    page_number=idx + 1,
                    image=pil_img,
                    png_bytes=png_bytes,
                    dpi=target_dpi,
                    page_sha256=compute_sha256(png_bytes),
                )
            )
        return rendered
    finally:
        pdf.close()


def render_image_pages(
    data: bytes,
    target_dpi: int = 300,
    max_pages: int = 25,
) -> list[RenderedPage]:
    """Decode single-frame or multi-frame (TIFF/WEBP) images into normalized RGB PNG pages."""
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:
        raise CorruptDocumentError(f"Failed to decode image payload: {exc}") from exc

    frames: list[Image.Image] = []
    for frame in ImageSequence.Iterator(img):
        frames.append(frame.convert("RGB"))
        if len(frames) > max_pages:
            raise PageLimitExceededError(
                f"Multi-frame image exceeds the maximum allowed pages ({max_pages})"
            )

    if not frames:
        raise CorruptDocumentError("Image contains 0 frames")

    rendered: list[RenderedPage] = []
    for idx, frame_rgb in enumerate(frames, start=1):
        png_bytes = _encode_png(frame_rgb, target_dpi)
        rendered.append(
            RenderedPage(
                page_number=idx,
                image=frame_rgb,
                png_bytes=png_bytes,
                dpi=target_dpi,
                page_sha256=compute_sha256(png_bytes),
            )
        )
    return rendered


def ingest_bytes(
    data: bytes,
    *,
    settings: Settings | None = None,
    storage_dir: Path | None = None,
    persist: bool = True,
) -> tuple[IngestResult, list[RenderedPage]]:
    """Validate, sniff, hash, split into pages, and optionally persist to content-addressed storage.

    Storage layout under `storage_dir/<doc_sha256[:2]>/<doc_sha256>/`:
    - `source.<ext>` — original raw bytes
    - `page_001.png`, `page_002.png`, ... — normalized raster pages
    """
    s = settings or get_settings()
    validate_upload_size(data, s.max_upload_bytes)
    mime = sniff_mime(data)
    doc_sha256 = compute_sha256(data)

    if mime == SupportedMime.PDF:
        rendered_pages = render_pdf_pages(data, target_dpi=s.target_dpi, max_pages=s.max_pages)
        ext = "pdf"
    else:
        rendered_pages = render_image_pages(data, target_dpi=s.target_dpi, max_pages=s.max_pages)
        ext = mime.value.split("/")[-1]

    target_root = storage_dir if storage_dir is not None else s.storage_dir
    doc_dir = target_root / doc_sha256[:2] / doc_sha256
    raw_file = doc_dir / f"source.{ext}"

    is_duplicate = raw_file.exists()
    raw_path_str: str | None = None

    page_assets: list[PageAsset] = []
    if persist:
        doc_dir.mkdir(parents=True, exist_ok=True)
        if not raw_file.exists():
            raw_file.write_bytes(data)
        raw_path_str = raw_file.as_posix()

    for rp in rendered_pages:
        img_path_str: str | None = None
        if persist:
            page_file = doc_dir / f"page_{rp.page_number:03d}.png"
            if not page_file.exists():
                page_file.write_bytes(rp.png_bytes)
            img_path_str = page_file.as_posix()

        page_assets.append(
            PageAsset(
                page_number=rp.page_number,
                width_px=rp.image.width,
                height_px=rp.image.height,
                dpi=rp.dpi,
                page_sha256=rp.page_sha256,
                image_path=img_path_str,
            )
        )

    result = IngestResult(
        doc_sha256=doc_sha256,
        mime_type=mime,
        size_bytes=len(data),
        page_count=len(page_assets),
        pages=page_assets,
        is_duplicate=is_duplicate,
        raw_path=raw_path_str,
    )
    return result, rendered_pages
