"""Unit and integration tests for the evaluation harness, metrics, and resumable JSONL cache."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from datagen import FaultType, LayoutStyle, generate_dataset, generate_sample
from docextract.extract import ground_extracted_document
from docextract.schemas import Invoice
from docextract.vision import extract_pdf_text_layer
from evals.metrics import score_extracted_fields
from evals.run_evals import (
    load_resume_cache,
    run_evaluation,
)
from evals.run_evals import (
    main as evals_cli_main,
)


def test_score_extracted_fields_exact_and_partial_match() -> None:
    sample = generate_sample(seed=1001, layout=LayoutStyle.CORPORATE_TABLE, fault=FaultType.NONE)
    assert isinstance(sample.gold_document, Invoice)
    layouts = extract_pdf_text_layer(sample.pdf_bytes)
    grounded = ground_extracted_document(sample.gold_document, layouts)

    # 1. Exact match achieves F1 = 1.0 and total_exact_match = True
    matched, n_pred, n_ref, prec, rec, f1, total_em, ground_cov = score_extracted_fields(
        grounded, sample.gold_document
    )
    assert matched == n_ref == n_pred
    assert prec == 1.0
    assert rec == 1.0
    assert f1 == 1.0
    assert total_em is True
    assert ground_cov >= 0.85

    # 2. Mutating `total` drops F1 below 1.0 and sets total_exact_match = False
    wrong_total_doc = grounded.model_copy(
        update={"total": grounded.total + Decimal("99.00")}
    )
    _, _, _, _, _, f1_bad, total_em_bad, _ = score_extracted_fields(
        wrong_total_doc, sample.gold_document
    )
    assert f1_bad < 1.0
    assert total_em_bad is False


@pytest.mark.asyncio
async def test_run_evaluation_offline_and_resume_cache(tmp_path: Path) -> None:
    samples = generate_dataset(count=8, out_dir=None, base_seed=1100)
    cache_file = tmp_path / "eval_cache.jsonl"

    # First run evaluates first 4 samples and writes them to cache_file
    summary_part, scores_part = await run_evaluation(
        samples[:4],
        offline=True,
        resume_path=cache_file,
    )
    assert summary_part.sample_count == 4
    assert len(scores_part) == 4
    assert len(load_resume_cache(cache_file)) == 4

    # Resuming across all 8 samples reuses the first 4 from cache and evaluates the remaining 4
    summary_full, scores_full = await run_evaluation(
        samples,
        offline=True,
        resume_path=cache_file,
    )
    assert summary_full.sample_count == 8
    assert len(scores_full) == 8
    assert summary_full.micro_f1 == 1.0
    assert summary_full.total_exact_match_rate == 1.0
    assert summary_full.fault_catch_rate == 1.0
    assert summary_full.clean_auto_approve_rate == 1.0


def test_evals_cli_writes_markdown_and_json_artifacts(tmp_path: Path) -> None:
    code = evals_cli_main(
        [
            "--count",
            "6",
            "--seed",
            "1200",
            "--offline",
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert code == 0
    assert (tmp_path / "latest.md").exists()
    assert (tmp_path / "latest.json").exists()
    md_text = (tmp_path / "latest.md").read_text(encoding="utf-8")
    assert "Field Micro F1 | 100.0%" in md_text
    assert "Fault-Injection Catch Rate | 100.0%" in md_text
