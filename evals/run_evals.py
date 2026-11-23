"""CLI evaluation runner with offline mode, resumable JSONL cache, and Markdown report."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections.abc import Sequence
from pathlib import Path

from datagen.generator import FaultType, GeneratedSample, LayoutStyle, generate_dataset
from docextract.config import Settings, get_settings
from docextract.extract.llm import ExtractionLLM, LiteLLMExtractorClient, ScriptedLLM
from docextract.extract.runner import extract_structured_document
from docextract.validate.rules import validate_document
from docextract.vision.pdf_text import extract_pdf_text_layer
from evals.metrics import (
    AggregateEvalMetrics,
    FieldScore,
    aggregate_scores,
    is_expected_fault_caught,
    score_extracted_fields,
)


def load_resume_cache(cache_path: Path | None) -> dict[str, FieldScore]:
    """Load previously completed `FieldScore` rows from a JSONL resume cache file."""
    if cache_path is None or not cache_path.exists():
        return {}
    cached: dict[str, FieldScore] = {}
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        item = FieldScore.model_validate_json(stripped)
        cached[item.sample_id] = item
    return cached


def append_resume_cache(cache_path: Path | None, score: FieldScore) -> None:
    """Append a single completed `FieldScore` row to the JSONL resume cache."""
    if cache_path is None:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as fh:
        fh.write(score.model_dump_json() + "\n")


async def evaluate_sample(
    sample: GeneratedSample,
    *,
    llm: ExtractionLLM,
    settings: Settings | None = None,
) -> FieldScore:
    """Run layout extraction, LLM extraction, bbox grounding, and validation on one sample."""
    s = settings or get_settings()
    t0 = time.perf_counter()
    layouts = extract_pdf_text_layer(sample.pdf_bytes)
    ext_out = await extract_structured_document(layouts, llm=llm, settings=s)
    val_report = validate_document(ext_out.document, settings=s)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    # Compare extracted values against `injected_document` (what is literally printed on the PDF)
    matched, n_pred, n_ref, prec, rec, f1, total_em, ground_cov = score_extracted_fields(
        ext_out.document,
        sample.injected_document,
    )
    fault_caught = is_expected_fault_caught(sample.fault, val_report.failed_rule_ids)

    return FieldScore(
        sample_id=sample.sample_id,
        layout=sample.layout.value,
        fault=sample.fault.value,
        matched_fields=matched,
        predicted_fields=n_pred,
        gold_fields=n_ref,
        precision=prec,
        recall=rec,
        f1=f1,
        total_exact_match=total_em,
        grounding_coverage=ground_cov,
        expected_fault_caught=fault_caught,
        validation_passed=val_report.passed,
        latency_ms=elapsed_ms,
        cost_usd=ext_out.total_cost_usd,
    )


async def run_evaluation(
    samples: Sequence[GeneratedSample],
    *,
    offline: bool = False,
    llm: ExtractionLLM | None = None,
    resume_path: Path | None = None,
    settings: Settings | None = None,
) -> tuple[AggregateEvalMetrics, list[FieldScore]]:
    """Evaluate all `samples`, reusing cached entries from `resume_path` when available."""
    s = settings or get_settings()
    cached = load_resume_cache(resume_path)
    scores: list[FieldScore] = []

    for sample in samples:
        if sample.sample_id in cached:
            scores.append(cached[sample.sample_id])
            continue

        if llm is not None:
            active_llm: ExtractionLLM = llm
        elif offline:
            active_llm = ScriptedLLM([sample.injected_document.model_dump_json()])
        else:
            active_llm = LiteLLMExtractorClient(model=s.llm_model, timeout_s=s.llm_timeout_s)

        score = await evaluate_sample(sample, llm=active_llm, settings=s)
        append_resume_cache(resume_path, score)
        scores.append(score)

    return aggregate_scores(scores), scores


def format_markdown_report(
    summary: AggregateEvalMetrics,
    scores: Sequence[FieldScore],
    *,
    mode_label: str = "Offline Deterministic Benchmark",
) -> str:
    """Format dataset-wide and per-layout evaluation metrics as a Markdown report."""
    by_layout: dict[str, list[FieldScore]] = {}
    for sc in scores:
        by_layout.setdefault(sc.layout, []).append(sc)

    lines: list[str] = [
        "# docextract Evaluation Benchmark Report",
        "",
        f"**Mode:** {mode_label}  ",
        f"**Total Samples Evaluated:** {summary.sample_count} "
        f"({summary.clean_sample_count} clean/scan, {summary.fault_sample_count} fault-injected)",
        "",
        "## Overall Pipeline Metrics",
        "",
        "| Metric | Score |",
        "|---|---|",
        f"| Field Micro Precision | {summary.micro_precision * 100:.1f}% |",
        f"| Field Micro Recall | {summary.micro_recall * 100:.1f}% |",
        f"| Field Micro F1 | {summary.micro_f1 * 100:.1f}% |",
        f"| Field Macro F1 | {summary.macro_f1 * 100:.1f}% |",
        f"| Grand Total Exact-Match | {summary.total_exact_match_rate * 100:.1f}% |",
        f"| Bounding-Box Grounding Coverage | {summary.mean_grounding_coverage * 100:.1f}% |",
        f"| Fault-Injection Catch Rate | {summary.fault_catch_rate * 100:.1f}% |",
        f"| Clean Auto-Approval Rate | {summary.clean_auto_approve_rate * 100:.1f}% |",
        f"| Mean Pipeline Latency | {summary.mean_latency_ms:.1f} ms |",
        "",
        "## Breakdown by Document Layout",
        "",
        "| Layout | Samples | Micro F1 | Total Exact-Match | Bbox Grounding | Fault Catch Rate |",
        "|---|---|---|---|---|---|",
    ]

    for layout in (
        LayoutStyle.CORPORATE_TABLE.value,
        LayoutStyle.MODERN_SIDEBAR.value,
        LayoutStyle.THERMAL_RECEIPT.value,
    ):
        group = by_layout.get(layout, [])
        if not group:
            continue
        agg = aggregate_scores(group)
        lines.append(
            f"| `{layout}` | {agg.sample_count} | {agg.micro_f1 * 100:.1f}% | "
            f"{agg.total_exact_match_rate * 100:.1f}% | "
            f"{agg.mean_grounding_coverage * 100:.1f}% | "
            f"{agg.fault_catch_rate * 100:.1f}% |"
        )

    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evals.run_evals",
        description="Run field-level extraction, grounding, and fault-catch evaluation benchmarks.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=24,
        help="Number of synthetic documents to evaluate (default: 24)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
        help="Base random seed for benchmark dataset generation",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run with deterministic ScriptedLLM (no live API key required)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Optional JSONL file path to cache and resume per-sample evaluation results",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("evals/results"),
        help="Directory to write latest.md and JSON summary artifacts",
    )
    args = parser.parse_args(argv)

    samples = generate_dataset(count=args.count, out_dir=None, base_seed=args.seed)
    summary, scores = asyncio.run(
        run_evaluation(
            samples,
            offline=args.offline,
            resume_path=args.resume,
        )
    )

    mode_label = (
        "Offline Deterministic Benchmark (ScriptedLLM + PDF Layout + Validation Engine)"
        if args.offline
        else f"Live LLM Evaluation ({get_settings().llm_model})"
    )
    md_report = format_markdown_report(summary, scores, mode_label=mode_label)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    md_path = args.out_dir / "latest.md"
    json_path = args.out_dir / "latest.json"
    md_path.write_text(md_report, encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "summary": summary.__dict__,
                "samples": [s.model_dump() for s in scores],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(md_report)
    return 0


__all__ = [
    "AggregateEvalMetrics",
    "FaultType",
    "FieldScore",
    "append_resume_cache",
    "evaluate_sample",
    "format_markdown_report",
    "load_resume_cache",
    "main",
    "run_evaluation",
]


if __name__ == "__main__":
    raise SystemExit(main())
