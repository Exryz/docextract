"""Unit tests for LLM structured extraction, validation repair loop, and bbox grounding."""

from __future__ import annotations

import json

import pytest

from datagen import FaultType, LayoutStyle, generate_sample
from docextract.config import Settings
from docextract.extract import (
    ExtractionFailedError,
    ScriptedLLM,
    detect_document_kind,
    extract_structured_document,
)
from docextract.schemas import DocumentKind, Invoice, Receipt
from docextract.vision import extract_pdf_text_layer


@pytest.mark.asyncio
async def test_extract_invoice_first_try_and_ground_bboxes() -> None:
    sample = generate_sample(seed=101, layout=LayoutStyle.CORPORATE_TABLE, fault=FaultType.NONE)
    assert isinstance(sample.gold_document, Invoice)
    layouts = extract_pdf_text_layer(sample.pdf_bytes)

    assert detect_document_kind(layouts) == DocumentKind.INVOICE

    # Wrap valid JSON in markdown code fences to verify fence stripping
    fenced_json = f"```json\n{sample.gold_document.model_dump_json(indent=2)}\n```"
    llm = ScriptedLLM([fenced_json])

    out = await extract_structured_document(layouts, llm=llm)
    assert out.attempts == 1
    assert out.repair_errors == []
    assert out.total_prompt_tokens == 120
    assert isinstance(out.document, Invoice)
    assert out.document.invoice_number == sample.gold_document.invoice_number
    assert out.document.total == sample.gold_document.total

    # Verify automatic bounding-box grounding attached page coordinates
    assert "invoice_number" in out.document.field_bboxes
    assert "total" in out.document.field_bboxes
    assert out.document.vendor.source_bbox is not None
    assert out.document.line_items[0].source_bbox is not None


@pytest.mark.asyncio
async def test_extract_repair_loop_recovers_from_invalid_json_and_schema_error() -> None:
    sample = generate_sample(seed=202, layout=LayoutStyle.THERMAL_RECEIPT, fault=FaultType.NONE)
    assert isinstance(sample.gold_document, Receipt)
    layouts = extract_pdf_text_layer(sample.pdf_bytes)

    assert detect_document_kind(layouts) == DocumentKind.RECEIPT

    # Attempt 1: truncated/malformed JSON
    bad_json = '{"kind": "receipt", "transaction_date": "not-a-date"'
    # Attempt 2: valid JSON but negative subtotal & empty line_items (fails Pydantic validation)
    invalid_schema_payload = json.loads(sample.gold_document.model_dump_json())
    invalid_schema_payload["subtotal"] = "-50.00"
    invalid_schema_payload["line_items"] = []
    # Attempt 3: valid gold JSON
    good_json = sample.gold_document.model_dump_json()

    llm = ScriptedLLM([bad_json, json.dumps(invalid_schema_payload), good_json])
    settings = Settings(max_repair_attempts=2)

    out = await extract_structured_document(layouts, llm=llm, settings=settings)
    assert out.attempts == 3
    assert len(out.repair_errors) == 2
    assert len(llm.calls) == 3
    # Verify the repair prompt included the validation error feedback
    last_call_messages = llm.calls[-1]
    assert any("failed schema/JSON validation" in m["content"] for m in last_call_messages)
    assert isinstance(out.document, Receipt)
    assert out.document.receipt_number == sample.gold_document.receipt_number
    assert "total" in out.document.field_bboxes


@pytest.mark.asyncio
async def test_extract_raises_after_exhausting_repair_attempts() -> None:
    sample = generate_sample(seed=303, layout=LayoutStyle.MODERN_SIDEBAR, fault=FaultType.NONE)
    layouts = extract_pdf_text_layer(sample.pdf_bytes)

    llm = ScriptedLLM(["not json 1", "not json 2"])
    settings = Settings(max_repair_attempts=1)

    with pytest.raises(ExtractionFailedError) as exc_info:
        await extract_structured_document(layouts, llm=llm, settings=settings)

    assert len(exc_info.value.errors) == 2
    assert len(exc_info.value.call_stats) == 2
