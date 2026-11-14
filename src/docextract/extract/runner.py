"""Structured extraction runner with validation-error repair loop and bounding-box grounding."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from docextract.config import Settings, get_settings
from docextract.extract.llm import ExtractionLLM, LiteLLMExtractorClient, LLMCallStats
from docextract.extract.prompts import (
    build_extraction_messages,
    build_repair_message,
    detect_document_kind,
)
from docextract.schemas.document import (
    DocumentKind,
    ExtractedDocument,
    Invoice,
    LineItem,
    Receipt,
    TaxLine,
)
from docextract.schemas.grounding import BoundingBox
from docextract.vision.pdf_text import PageLayout

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


class ExtractionFailedError(RuntimeError):
    """Raised when the LLM fails to produce a schema-valid document after all repair attempts."""

    def __init__(self, message: str, errors: list[str], call_stats: list[LLMCallStats]) -> None:
        super().__init__(message)
        self.errors = errors
        self.call_stats = call_stats


@dataclass(frozen=True)
class ExtractionOutput:
    """Validated, bbox-grounded `Invoice` or `Receipt` plus repair and token telemetry."""

    document: ExtractedDocument
    attempts: int
    repair_errors: list[str] = field(default_factory=list)
    call_stats: list[LLMCallStats] = field(default_factory=list)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(s.prompt_tokens for s in self.call_stats)

    @property
    def total_completion_tokens(self) -> int:
        return sum(s.completion_tokens for s in self.call_stats)

    @property
    def total_cost_usd(self) -> float:
        return round(sum(s.cost_usd for s in self.call_stats), 6)

    @property
    def total_latency_ms(self) -> float:
        return round(sum(s.latency_ms for s in self.call_stats), 2)


def _strip_json_fences(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = _JSON_FENCE_RE.sub("", cleaned).strip()
    return cleaned


def _find_bbox_across_pages(layouts: Sequence[PageLayout], text: str | None) -> BoundingBox | None:
    if not text or not text.strip():
        return None
    for page in layouts:
        bbox = page.find_matching_bbox(text)
        if bbox is not None:
            return bbox
    return None


def ground_extracted_document(
    doc: ExtractedDocument,
    layouts: Sequence[PageLayout],
) -> ExtractedDocument:
    """Attach `source_bbox` and `field_bboxes` by matching values against `PageLayout` words."""
    if not layouts:
        return doc

    field_bboxes: dict[str, BoundingBox] = dict(doc.field_bboxes)

    def register_field(name: str, candidate_text: str | None) -> BoundingBox | None:
        if name in field_bboxes:
            return field_bboxes[name]
        bbox = _find_bbox_across_pages(layouts, candidate_text)
        if bbox is not None:
            field_bboxes[name] = bbox
        return bbox

    grounded_items: list[LineItem] = []
    for idx, item in enumerate(doc.line_items):
        item_bbox = (
            item.source_bbox
            or _find_bbox_across_pages(layouts, item.description)
            or _find_bbox_across_pages(layouts, str(item.amount))
        )
        if item_bbox is not None:
            field_bboxes[f"line_items[{idx}]"] = item_bbox
            grounded_items.append(item.model_copy(update={"source_bbox": item_bbox}))
        else:
            grounded_items.append(item)

    grounded_taxes: list[TaxLine] = []
    for idx, tax in enumerate(doc.taxes):
        tax_bbox = (
            tax.source_bbox
            or _find_bbox_across_pages(layouts, str(tax.amount))
            or _find_bbox_across_pages(layouts, tax.label)
        )
        if tax_bbox is not None:
            field_bboxes[f"taxes[{idx}]"] = tax_bbox
            grounded_taxes.append(tax.model_copy(update={"source_bbox": tax_bbox}))
        else:
            grounded_taxes.append(tax)

    if isinstance(doc, Invoice):
        register_field("invoice_number", doc.invoice_number)
        register_field("issue_date", doc.issue_date.isoformat())
        if doc.due_date is not None:
            register_field("due_date", doc.due_date.isoformat())
        if doc.purchase_order is not None:
            register_field("purchase_order", doc.purchase_order)
        register_field("subtotal", str(doc.subtotal))
        register_field("tax_total", str(doc.tax_total))
        register_field("total", str(doc.total))

        vendor_bbox = doc.vendor.source_bbox or register_field("vendor.name", doc.vendor.name)
        vendor = (
            doc.vendor.model_copy(update={"source_bbox": vendor_bbox})
            if vendor_bbox is not None
            else doc.vendor
        )

        customer = doc.customer
        if customer is not None:
            cust_bbox = customer.source_bbox or register_field("customer.name", customer.name)
            if cust_bbox is not None:
                customer = customer.model_copy(update={"source_bbox": cust_bbox})

        return doc.model_copy(
            update={
                "vendor": vendor,
                "customer": customer,
                "line_items": grounded_items,
                "taxes": grounded_taxes,
                "field_bboxes": field_bboxes,
            }
        )

    assert isinstance(doc, Receipt)
    if doc.receipt_number is not None:
        register_field("receipt_number", doc.receipt_number)
    register_field("transaction_date", doc.transaction_date.isoformat())
    register_field("subtotal", str(doc.subtotal))
    register_field("tax_total", str(doc.tax_total))
    register_field("tip_total", str(doc.tip_total))
    register_field("total", str(doc.total))

    merch_bbox = doc.merchant.source_bbox or register_field("merchant.name", doc.merchant.name)
    merchant = (
        doc.merchant.model_copy(update={"source_bbox": merch_bbox})
        if merch_bbox is not None
        else doc.merchant
    )

    return doc.model_copy(
        update={
            "merchant": merchant,
            "line_items": grounded_items,
            "taxes": grounded_taxes,
            "field_bboxes": field_bboxes,
        }
    )


def _parse_and_validate(raw_text: str, kind: DocumentKind) -> ExtractedDocument:
    cleaned = _strip_json_fences(raw_text)
    payload: Any = json.loads(cleaned)
    if not isinstance(payload, dict):
        raise ValueError("Top-level JSON value must be an object")

    if kind == DocumentKind.INVOICE:
        return Invoice.model_validate(payload)
    return Receipt.model_validate(payload)


async def extract_structured_document(
    layouts: Sequence[PageLayout],
    *,
    llm: ExtractionLLM | None = None,
    kind: DocumentKind | None = None,
    settings: Settings | None = None,
) -> ExtractionOutput:
    """Run LLM extraction against `layouts` with validation-error repair and bbox grounding."""
    s = settings or get_settings()
    client = llm or LiteLLMExtractorClient(model=s.llm_model, timeout_s=s.llm_timeout_s)
    resolved_kind = detect_document_kind(layouts, hint=kind)

    messages = build_extraction_messages(layouts, kind=resolved_kind)
    max_attempts = 1 + max(0, s.max_repair_attempts)

    repair_errors: list[str] = []
    call_stats: list[LLMCallStats] = []

    for attempt in range(1, max_attempts + 1):
        completion = await client.complete(
            messages,
            temperature=s.llm_temperature,
            json_mode=True,
        )
        call_stats.append(completion.stats)

        try:
            doc = _parse_and_validate(completion.content, resolved_kind)
            grounded = ground_extracted_document(doc, layouts)
            return ExtractionOutput(
                document=grounded,
                attempts=attempt,
                repair_errors=repair_errors,
                call_stats=call_stats,
            )
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            err_msg = str(exc)
            repair_errors.append(err_msg)
            if attempt < max_attempts:
                messages.extend(build_repair_message(completion.content, err_msg))

    raise ExtractionFailedError(
        f"Extraction failed after {max_attempts} attempts: {repair_errors[-1]}",
        errors=repair_errors,
        call_stats=call_stats,
    )
