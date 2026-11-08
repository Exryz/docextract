"""OCR engine protocol, RapidOCR ONNX adapter, and hybrid PDF/OCR layout extractor."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from PIL import Image

from docextract.ingest.sniffer import SupportedMime, sniff_mime
from docextract.ingest.splitter import render_image_pages, render_pdf_pages
from docextract.schemas.grounding import BoundingBox, WordToken
from docextract.vision.pdf_text import PageLayout, extract_pdf_text_layer, sort_words_reading_order
from docextract.vision.preprocess import preprocess_page_image


class OcrEngine(Protocol):
    """Protocol for pluggable OCR engines (RapidOCR ONNX in production, fake in unit tests)."""

    def recognize(self, image: Image.Image, page_number: int) -> list[WordToken]: ...


class RapidOcrEngine:
    """CPU-only ONNX OCR engine backed by `rapidocr-onnxruntime`."""

    def __init__(self) -> None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:
            raise ImportError(
                "rapidocr-onnxruntime is required for OCR fallback. "
                "Install it with `pip install -e .[ocr]`."
            ) from exc
        self._engine: Any = RapidOCR()

    def recognize(self, image: Image.Image, page_number: int) -> list[WordToken]:
        import numpy as np

        rgb = image.convert("RGB") if image.mode != "RGB" else image
        arr = np.asarray(rgb)
        raw_results, _ = self._engine(arr)
        if not raw_results:
            return []

        w_px, h_px = float(rgb.width), float(rgb.height)
        tokens: list[WordToken] = []
        for item in raw_results:
            # item format: [ [[x0,y0], [x1,y0], [x1,y1], [x0,y1]], text, confidence ]
            quad, text, conf = item[0], str(item[1]).strip(), float(item[2])
            if not text:
                continue
            xs = [ float(pt[0]) for pt in quad ]
            ys = [ float(pt[1]) for pt in quad ]
            x0, x1 = max(0.0, min(xs)), min(w_px, max(xs))
            y0, y1 = max(0.0, min(ys)), min(h_px, max(ys))

            words = text.split()
            if len(words) <= 1:
                bbox = BoundingBox.from_pixel_rect(page_number, x0, y0, x1, y1, w_px, h_px)
                tokens.append(
                    WordToken(text=text, bbox=bbox, confidence=max(0.0, min(1.0, conf)))
                )
            else:
                # Subdivide multi-word OCR line box proportionally by character width
                total_chars = max(1, sum(len(w) for w in words))
                span_w = max(1.0, x1 - x0)
                cursor_x = x0
                for word in words:
                    word_w = span_w * (len(word) / total_chars)
                    wx1 = min(x1, cursor_x + word_w)
                    bbox = BoundingBox.from_pixel_rect(
                        page_number, cursor_x, y0, wx1, y1, w_px, h_px
                    )
                    tokens.append(
                        WordToken(text=word, bbox=bbox, confidence=max(0.0, min(1.0, conf)))
                    )
                    cursor_x = wx1

        return sort_words_reading_order(tokens)


class StaticOcrEngine:
    """Deterministic test double that returns pre-configured WordTokens per page."""

    def __init__(self, tokens_by_page: dict[int, Sequence[WordToken]] | None = None) -> None:
        self._tokens = {k: list(v) for k, v in (tokens_by_page or {}).items()}
        self.calls: list[int] = []

    def recognize(self, image: Image.Image, page_number: int) -> list[WordToken]:
        self.calls.append(page_number)
        return sort_words_reading_order(self._tokens.get(page_number, []))


def extract_document_layout(
    data: bytes,
    *,
    ocr_engine: OcrEngine | None = None,
    min_native_words_per_page: int = 3,
    target_dpi: int = 300,
    deskew_scans: bool = True,
) -> list[PageLayout]:
    """Extract `PageLayout` for every page in `data`.

    - For PDFs: extracts native PDF text layer first. Any page with `< min_native_words_per_page`
      words (e.g. a scanned image-only PDF page) automatically falls back to preprocessing + OCR.
    - For images (PNG/JPEG/WEBP/TIFF): runs preprocessing (deskew) + OCR on every page.
    """
    mime = sniff_mime(data)

    if mime == SupportedMime.PDF:
        native_layouts = extract_pdf_text_layer(data)
        needs_ocr = any(len(layout.words) < min_native_words_per_page for layout in native_layouts)
        if not needs_ocr:
            return native_layouts

        engine = ocr_engine or RapidOcrEngine()
        rendered_pages = render_pdf_pages(data, target_dpi=target_dpi)
        merged_layouts: list[PageLayout] = []
        for layout, rp in zip(native_layouts, rendered_pages, strict=True):
            if len(layout.words) >= min_native_words_per_page:
                merged_layouts.append(layout)
            else:
                prep = preprocess_page_image(rp.image, deskew=deskew_scans)
                ocr_words = engine.recognize(prep.image, page_number=layout.page_number)
                merged_layouts.append(
                    PageLayout(
                        page_number=layout.page_number,
                        width_pt=layout.width_pt,
                        height_pt=layout.height_pt,
                        words=ocr_words,
                        source="ocr",
                        skew_angle_deg=prep.skew_angle_deg,
                    )
                )
        return merged_layouts

    # Raster image upload -> always preprocess + OCR
    engine = ocr_engine or RapidOcrEngine()
    rendered_pages = render_image_pages(data, target_dpi=target_dpi)
    image_layouts: list[PageLayout] = []
    for rp in rendered_pages:
        prep = preprocess_page_image(rp.image, deskew=deskew_scans)
        ocr_words = engine.recognize(prep.image, page_number=rp.page_number)
        image_layouts.append(
            PageLayout(
                page_number=rp.page_number,
                width_pt=float(prep.image.width),
                height_pt=float(prep.image.height),
                words=ocr_words,
                source="ocr",
                skew_angle_deg=prep.skew_angle_deg,
            )
        )
    return image_layouts
