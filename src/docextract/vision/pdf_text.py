"""Native PDF text and word bounding-box extraction using `pypdfium2`."""

from __future__ import annotations

import re

import pypdfium2 as pdfium
from pydantic import BaseModel, ConfigDict, Field

from docextract.schemas.grounding import BoundingBox, WordToken

_WHITESPACE_RE = re.compile(r"\s+")


class PageLayout(BaseModel):
    """Extracted word tokens, reading-order text, and source provenance for a single page."""

    model_config = ConfigDict(frozen=True)

    page_number: int = Field(ge=1)
    width_pt: float = Field(gt=0.0)
    height_pt: float = Field(gt=0.0)
    words: list[WordToken] = Field(default_factory=list)
    source: str = Field(default="pdf_text", description="'pdf_text' or 'ocr'")
    skew_angle_deg: float = 0.0

    @property
    def full_text(self) -> str:
        """Reconstruct reading-order text from word tokens grouped by approximate visual lines."""
        if not self.words:
            return ""
        sorted_words = sort_words_reading_order(self.words)
        lines: list[list[WordToken]] = []
        for w in sorted_words:
            if not lines:
                lines.append([w])
                continue
            prev_line = lines[-1]
            prev_cy = sum((tok.bbox.y0 + tok.bbox.y1) * 0.5 for tok in prev_line) / len(prev_line)
            prev_h = max(0.008, sum(tok.bbox.height for tok in prev_line) / len(prev_line))
            curr_cy = (w.bbox.y0 + w.bbox.y1) * 0.5
            if abs(curr_cy - prev_cy) <= prev_h * 0.65:
                prev_line.append(w)
            else:
                lines.append([w])

        rendered_lines: list[str] = []
        for line in lines:
            line_sorted = sorted(line, key=lambda tok: tok.bbox.x0)
            rendered_lines.append(" ".join(tok.text for tok in line_sorted))
        return "\n".join(rendered_lines)

    def find_matching_bbox(self, target_text: str) -> BoundingBox | None:
        """Find the bounding box of a token or contiguous multi-word span matching `target_text`."""
        cleaned = _WHITESPACE_RE.sub(" ", target_text.strip()).lower()
        if not cleaned or not self.words:
            return None

        target_parts = cleaned.split(" ")
        ordered = sort_words_reading_order(self.words)
        norm_tokens = [tok.text.strip().lower().strip(",:;()") for tok in ordered]

        n = len(target_parts)
        for i in range(len(ordered) - n + 1):
            window = norm_tokens[i : i + n]
            if window == target_parts:
                return BoundingBox.merge([tok.bbox for tok in ordered[i : i + n]])

        # Fallback: substring match inside a single token (e.g. "$1,240.00" vs "1,240.00")
        stripped_target = cleaned.lstrip("$€£₱¥")
        for tok in ordered:
            tok_clean = tok.text.strip().lower().lstrip("$€£₱¥").rstrip(",.")
            if stripped_target and tok_clean == stripped_target:
                return tok.bbox
        return None


def sort_words_reading_order(words: list[WordToken]) -> list[WordToken]:
    """Sort tokens top-to-bottom into horizontal bands, then left-to-right within each band."""
    if len(words) <= 1:
        return list(words)

    by_y = sorted(words, key=lambda w: (w.bbox.y0 + w.bbox.y1) * 0.5)
    bands: list[list[WordToken]] = []
    for w in by_y:
        if not bands:
            bands.append([w])
            continue
        band = bands[-1]
        band_cy = sum((t.bbox.y0 + t.bbox.y1) * 0.5 for t in band) / len(band)
        band_h = max(0.008, sum(t.bbox.height for t in band) / len(band))
        w_cy = (w.bbox.y0 + w.bbox.y1) * 0.5
        if abs(w_cy - band_cy) <= band_h * 0.65:
            band.append(w)
        else:
            bands.append([w])

    result: list[WordToken] = []
    for band in bands:
        result.extend(sorted(band, key=lambda t: t.bbox.x0))
    return result


def _extract_page_words(
    page: pdfium.PdfPage,
    page_number: int,
) -> tuple[float, float, list[WordToken]]:
    width_pt, height_pt = page.get_size()
    textpage = page.get_textpage()
    try:
        n_chars = textpage.count_chars()
        if n_chars <= 0 or width_pt <= 0 or height_pt <= 0:
            return width_pt, height_pt, []

        words: list[WordToken] = []
        current_chars: list[str] = []
        current_boxes: list[tuple[float, float, float, float]] = []

        def flush_word() -> None:
            if not current_chars or not current_boxes:
                current_chars.clear()
                current_boxes.clear()
                return
            word_str = "".join(current_chars).strip()
            if word_str:
                # PDF coordinate origin is bottom-left; convert to top-left normalized [0, 1]
                left = min(b[0] for b in current_boxes)
                bottom = min(b[1] for b in current_boxes)
                right = max(b[2] for b in current_boxes)
                top = max(b[3] for b in current_boxes)

                nx0 = max(0.0, min(1.0, left / width_pt))
                ny0 = max(0.0, min(1.0, (height_pt - top) / height_pt))
                nx1 = max(nx0, min(1.0, right / width_pt))
                ny1 = max(ny0, min(1.0, (height_pt - bottom) / height_pt))

                words.append(
                    WordToken(
                        text=word_str,
                        bbox=BoundingBox(
                            page=page_number,
                            x0=round(nx0, 6),
                            y0=round(ny0, 6),
                            x1=round(nx1, 6),
                            y1=round(ny1, 6),
                        ),
                        confidence=1.0,
                    )
                )
            current_chars.clear()
            current_boxes.clear()

        for idx in range(n_chars):
            ch = textpage.get_text_range(idx, 1)
            if not ch or ch.isspace() or ch in ("\x00", "\ufffe", "\uffff"):
                flush_word()
                continue
            rect = textpage.get_charbox(idx, loose=True)
            current_chars.append(ch)
            current_boxes.append(rect)

        flush_word()
        return width_pt, height_pt, sort_words_reading_order(words)
    finally:
        textpage.close()


def extract_pdf_text_layer(data: bytes) -> list[PageLayout]:
    """Extract word tokens and normalized bounding boxes from each page of a PDF document."""
    pdf = pdfium.PdfDocument(data)
    try:
        layouts: list[PageLayout] = []
        for idx in range(len(pdf)):
            page = pdf[idx]
            try:
                w_pt, h_pt, words = _extract_page_words(page, page_number=idx + 1)
                layouts.append(
                    PageLayout(
                        page_number=idx + 1,
                        width_pt=w_pt,
                        height_pt=h_pt,
                        words=words,
                        source="pdf_text",
                    )
                )
            finally:
                page.close()
        return layouts
    finally:
        pdf.close()
