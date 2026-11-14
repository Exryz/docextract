"""Prompt builders for initial schema-constrained extraction and validation-error repair."""

from __future__ import annotations

import json
from collections.abc import Sequence

from docextract.extract.llm import Message
from docextract.schemas.document import DocumentKind, Invoice, Receipt
from docextract.vision.pdf_text import PageLayout

_SYSTEM_PROMPT = """You are a precise financial document extraction engine.
Extract the structured fields from the provided document pages into a single valid JSON object
matching the target JSON Schema.

Rules:
1. Output ONLY raw valid JSON (no Markdown fences, no commentary).
2. Monetary amounts must be exact decimals (e.g. "1240.00") without currency symbols.
3. Dates must be formatted as ISO 8601 "YYYY-MM-DD".
4. Preserve every line item in order with its exact description, quantity, unit_price, and amount.
5. Do not invent values that are not present in the document text."""


def format_layout_context(layouts: Sequence[PageLayout]) -> str:
    """Render page text and word-level bounding boxes into a compact prompt context."""
    sections: list[str] = []
    for page in layouts:
        header = f"=== PAGE {page.page_number} (source={page.source}) ==="
        sections.append(f"{header}\n{page.full_text}")
    return "\n\n".join(sections)


def detect_document_kind(
    layouts: Sequence[PageLayout], hint: DocumentKind | None = None
) -> DocumentKind:
    """Heuristically classify whether pages represent an `invoice` or `receipt`."""
    if hint is not None:
        return hint
    combined = " ".join(p.full_text.lower() for p in layouts)
    if "invoice" in combined or "bill to" in combined or "purchase order" in combined:
        return DocumentKind.INVOICE
    if "receipt" in combined or "tip:" in combined or "paid via card" in combined:
        return DocumentKind.RECEIPT
    return DocumentKind.INVOICE


def build_extraction_messages(
    layouts: Sequence[PageLayout],
    kind: DocumentKind = DocumentKind.INVOICE,
) -> list[Message]:
    """Build the initial system + user messages with embedded JSON schema and layout text."""
    schema_cls = Invoice if kind == DocumentKind.INVOICE else Receipt
    schema_json = json.dumps(schema_cls.model_json_schema(), indent=2)
    doc_context = format_layout_context(layouts)

    user_content = (
        f"Target document kind: `{kind.value}`\n\n"
        f"JSON Schema:\n{schema_json}\n\n"
        f"Document Text:\n{doc_context}\n\n"
        "Return the extracted JSON object:"
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_repair_message(raw_output: str, error_summary: str) -> list[Message]:
    """Append the invalid assistant output and a repair instruction with validation errors."""
    return [
        {"role": "assistant", "content": raw_output},
        {
            "role": "user",
            "content": (
                "Your previous output failed schema/JSON validation with the following errors:\n"
                f"{error_summary}\n\n"
                "Fix the errors and return ONLY the corrected JSON object conforming to the schema."
            ),
        },
    ]
