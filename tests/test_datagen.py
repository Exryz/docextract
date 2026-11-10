"""Unit tests for the synthetic invoice/receipt generator and fault injector."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from datagen import (
    FaultType,
    LayoutStyle,
    generate_dataset,
    generate_sample,
)
from datagen.cli import main as datagen_cli_main
from docextract.schemas import Invoice, Receipt
from docextract.vision import deskew_image, extract_pdf_text_layer


def test_generate_clean_samples_across_all_layouts() -> None:
    for layout in (
        LayoutStyle.CORPORATE_TABLE,
        LayoutStyle.MODERN_SIDEBAR,
        LayoutStyle.THERMAL_RECEIPT,
    ):
        sample = generate_sample(seed=42, layout=layout, fault=FaultType.NONE)
        assert sample.gold_document == sample.injected_document

        layouts = extract_pdf_text_layer(sample.pdf_bytes)
        assert len(layouts) == 1
        full_text = layouts[0].full_text

        if isinstance(sample.gold_document, Invoice):
            assert sample.gold_document.invoice_number in full_text
            assert str(sample.gold_document.total) in full_text
            assert sample.gold_document.computed_subtotal == sample.gold_document.subtotal
            assert sample.gold_document.computed_total == sample.gold_document.total
            # Verify bbox lookup finds the invoice number in the rendered PDF
            bbox = layouts[0].find_matching_bbox(sample.gold_document.invoice_number)
            assert bbox is not None
        else:
            assert isinstance(sample.gold_document, Receipt)
            assert sample.gold_document.receipt_number is not None
            assert sample.gold_document.receipt_number in full_text
            assert str(sample.gold_document.total) in full_text
            assert sample.gold_document.computed_total == sample.gold_document.total


def test_fault_injection_mutates_target_fields_while_preserving_gold() -> None:
    s_line = generate_sample(
        seed=7, layout=LayoutStyle.CORPORATE_TABLE, fault=FaultType.WRONG_LINE_AMOUNT
    )
    assert isinstance(s_line.gold_document, Invoice)
    assert isinstance(s_line.injected_document, Invoice)
    assert s_line.injected_document.line_items[0].amount == (
        s_line.gold_document.line_items[0].amount + Decimal("25.00")
    )
    assert s_line.injected_document.computed_subtotal != s_line.injected_document.subtotal

    s_total = generate_sample(
        seed=8, layout=LayoutStyle.MODERN_SIDEBAR, fault=FaultType.WRONG_TOTAL
    )
    assert isinstance(s_total.injected_document, Invoice)
    assert s_total.injected_document.total != s_total.injected_document.computed_total

    s_dates = generate_sample(
        seed=9, layout=LayoutStyle.CORPORATE_TABLE, fault=FaultType.INVERTED_DATES
    )
    assert isinstance(s_dates.injected_document, Invoice)
    assert s_dates.injected_document.due_date is not None
    assert s_dates.injected_document.due_date < s_dates.injected_document.issue_date


def test_rotated_scan_fault_produces_deskewable_png() -> None:
    sample = generate_sample(
        seed=15, layout=LayoutStyle.CORPORATE_TABLE, fault=FaultType.ROTATED_SCAN
    )
    assert sample.scan_png_bytes is not None
    import io

    from PIL import Image

    scan_img = Image.open(io.BytesIO(sample.scan_png_bytes))
    _, detected_angle, was_deskewed = deskew_image(scan_img)
    assert was_deskewed is True
    assert 1.5 <= abs(detected_angle) <= 5.5


def test_generate_dataset_and_cli_persistence(tmp_path: Path) -> None:
    exit_code = datagen_cli_main(["--count", "6", "--out-dir", str(tmp_path), "--seed", "200"])
    assert exit_code == 0

    gold_files = sorted(tmp_path.glob("*.gold.json"))
    pdf_files = sorted(tmp_path.glob("*.pdf"))
    assert len(gold_files) == 6
    assert len(pdf_files) == 6

    payload = json.loads(gold_files[0].read_text(encoding="utf-8"))
    assert "sample_id" in payload
    assert "gold_document" in payload
    assert "injected_document" in payload

    samples = generate_dataset(count=4, out_dir=None, base_seed=300)
    assert len(samples) == 4
