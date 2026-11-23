"""Field-level Precision/Recall/F1, total exact-match, grounding rate, and fault-catch metrics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from datagen.generator import FaultType
from docextract.schemas.document import ExtractedDocument
from docextract.scoring.confidence import extract_key_fields


class FieldScore(BaseModel):
    """Per-sample field-level comparison between predicted and gold documents."""

    model_config = ConfigDict(frozen=True)

    sample_id: str
    layout: str
    fault: str
    matched_fields: int
    predicted_fields: int
    gold_fields: int
    precision: float
    recall: float
    f1: float
    total_exact_match: bool
    grounding_coverage: float
    expected_fault_caught: bool
    validation_passed: bool
    latency_ms: float
    cost_usd: float


@dataclass(frozen=True)
class AggregateEvalMetrics:
    """Dataset-wide evaluation summary across all evaluated document samples."""

    sample_count: int
    clean_sample_count: int
    fault_sample_count: int
    micro_precision: float
    micro_recall: float
    micro_f1: float
    macro_f1: float
    total_exact_match_rate: float
    mean_grounding_coverage: float
    fault_catch_rate: float
    clean_auto_approve_rate: float
    mean_latency_ms: float
    total_cost_usd: float


def _normalize_val(val: str) -> str:
    return " ".join(val.strip().lower().split())


def score_extracted_fields(
    predicted: ExtractedDocument,
    reference: ExtractedDocument,
) -> tuple[int, int, int, float, float, float, bool, float]:
    """Compare key fields of `predicted` against `reference`.

    Returns `(matched, n_pred, n_ref, prec, rec, f1, total_exact_match, grounding_coverage)`.
    """
    pred_map = {
        k: _normalize_val(v) for k, v in extract_key_fields(predicted).items() if v.strip()
    }
    ref_map = {
        k: _normalize_val(v) for k, v in extract_key_fields(reference).items() if v.strip()
    }

    matched = sum(1 for k, ref_v in ref_map.items() if pred_map.get(k) == ref_v)
    n_pred = len(pred_map)
    n_ref = len(ref_map)

    precision = round(matched / n_pred, 4) if n_pred > 0 else 0.0
    recall = round(matched / n_ref, 4) if n_ref > 0 else 0.0
    f1 = (
        round((2 * precision * recall) / (precision + recall), 4)
        if (precision + recall) > 0
        else 0.0
    )

    total_exact = pred_map.get("total") == ref_map.get("total")

    # Grounding coverage: fraction of predicted key fields with an attached page BoundingBox
    grounded_count = 0
    for k in pred_map:
        is_grounded = (
            k in predicted.field_bboxes
            or (k.startswith("line_items[") and k.split(".")[0] in predicted.field_bboxes)
            or (k.startswith("taxes[") and k.split(".")[0] in predicted.field_bboxes)
            or (k.startswith("vendor.") and "vendor.name" in predicted.field_bboxes)
            or (k.startswith("customer.") and "customer.name" in predicted.field_bboxes)
            or (k.startswith("merchant.") and "merchant.name" in predicted.field_bboxes)
        )
        if is_grounded:
            grounded_count += 1

    grounding_cov = round(grounded_count / n_pred, 4) if n_pred > 0 else 0.0
    return matched, n_pred, n_ref, precision, recall, f1, total_exact, grounding_cov


def is_expected_fault_caught(fault: FaultType, failed_rule_ids: Sequence[str]) -> bool:
    """Return True if the validation engine triggered the rule corresponding to `fault`."""
    expected_rule_by_fault: dict[FaultType, str] = {
        FaultType.WRONG_LINE_AMOUNT: "line_item_math",
        FaultType.WRONG_SUBTOTAL: "subtotal_math",
        FaultType.WRONG_TAX: "tax_total_math",
        FaultType.WRONG_TOTAL: "grand_total_math",
        FaultType.INVERTED_DATES: "invoice_due_date_order",
    }
    target_rule = expected_rule_by_fault.get(fault)
    if target_rule is None:
        # Clean or rotated_scan samples have no mathematical fault; passing validation is expected
        return len(failed_rule_ids) == 0
    return target_rule in failed_rule_ids


def aggregate_scores(scores: Sequence[FieldScore]) -> AggregateEvalMetrics:
    """Aggregate per-sample `FieldScore`s into dataset-level Precision/Recall/F1 and catch rates."""
    if not scores:
        return AggregateEvalMetrics(
            sample_count=0,
            clean_sample_count=0,
            fault_sample_count=0,
            micro_precision=0.0,
            micro_recall=0.0,
            micro_f1=0.0,
            macro_f1=0.0,
            total_exact_match_rate=0.0,
            mean_grounding_coverage=0.0,
            fault_catch_rate=0.0,
            clean_auto_approve_rate=0.0,
            mean_latency_ms=0.0,
            total_cost_usd=0.0,
        )

    total_matched = sum(s.matched_fields for s in scores)
    total_pred = sum(s.predicted_fields for s in scores)
    total_gold = sum(s.gold_fields for s in scores)

    micro_p = round(total_matched / total_pred, 4) if total_pred > 0 else 0.0
    micro_r = round(total_matched / total_gold, 4) if total_gold > 0 else 0.0
    micro_f1 = (
        round((2 * micro_p * micro_r) / (micro_p + micro_r), 4)
        if (micro_p + micro_r) > 0
        else 0.0
    )
    macro_f1 = round(sum(s.f1 for s in scores) / len(scores), 4)
    total_em = round(sum(1 for s in scores if s.total_exact_match) / len(scores), 4)
    mean_ground = round(sum(s.grounding_coverage for s in scores) / len(scores), 4)

    clean_samples = [
        s for s in scores if s.fault in (FaultType.NONE.value, FaultType.ROTATED_SCAN.value)
    ]
    fault_samples = [
        s for s in scores if s.fault not in (FaultType.NONE.value, FaultType.ROTATED_SCAN.value)
    ]

    fault_catch = (
        round(sum(1 for s in fault_samples if s.expected_fault_caught) / len(fault_samples), 4)
        if fault_samples
        else 1.0
    )
    clean_approve = (
        round(sum(1 for s in clean_samples if s.validation_passed) / len(clean_samples), 4)
        if clean_samples
        else 1.0
    )

    mean_lat = round(sum(s.latency_ms for s in scores) / len(scores), 2)
    tot_cost = round(sum(s.cost_usd for s in scores), 6)

    return AggregateEvalMetrics(
        sample_count=len(scores),
        clean_sample_count=len(clean_samples),
        fault_sample_count=len(fault_samples),
        micro_precision=micro_p,
        micro_recall=micro_r,
        micro_f1=micro_f1,
        macro_f1=macro_f1,
        total_exact_match_rate=total_em,
        mean_grounding_coverage=mean_ground,
        fault_catch_rate=fault_catch,
        clean_auto_approve_rate=clean_approve,
        mean_latency_ms=mean_lat,
        total_cost_usd=tot_cost,
    )
