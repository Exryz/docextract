"""Field and document confidence scoring fusing OCR, bbox grounding, consistency, and rules."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from docextract.config import Settings, get_settings
from docextract.schemas.document import ExtractedDocument, Invoice, Receipt
from docextract.schemas.grounding import BoundingBox
from docextract.validate.rules import Severity, ValidationReport
from docextract.vision.pdf_text import PageLayout


class FieldConfidence(BaseModel):
    """Confidence breakdown for a single extracted field path."""

    model_config = ConfigDict(frozen=True)

    field_path: str
    value_str: str
    ocr_score: float = Field(
        ge=0.0, le=1.0, description="Mean confidence of underlying page tokens"
    )
    grounding_score: float = Field(
        ge=0.0, le=1.0, description="1.0 if matched to a page bounding box, 0.4 if ungrounded"
    )
    consistency_score: float = Field(
        ge=0.0, le=1.0, description="Fraction of self-consistency samples agreeing on this value"
    )
    rule_score: float = Field(
        ge=0.0,
        le=1.0,
        description="1.0 if all rules on this field pass, 0.5 on warning, 0.15 on error",
    )
    combined_score: float = Field(ge=0.0, le=1.0)
    needs_review: bool = False


class ConfidenceReport(BaseModel):
    """Aggregated confidence report across all extracted fields on a document."""

    model_config = ConfigDict(frozen=True)

    document_confidence: float = Field(ge=0.0, le=1.0)
    min_field_confidence: float = Field(ge=0.0, le=1.0)
    needs_review: bool
    low_confidence_fields: list[str] = Field(default_factory=list)
    fields: dict[str, FieldConfidence] = Field(default_factory=dict)


def extract_key_fields(doc: ExtractedDocument) -> dict[str, str]:
    """Flatten key scalar and line-item fields from `doc` into `{field_path: string_value}`."""
    fields: dict[str, str] = {}
    if isinstance(doc, Invoice):
        fields["invoice_number"] = doc.invoice_number
        fields["issue_date"] = doc.issue_date.isoformat()
        if doc.due_date is not None:
            fields["due_date"] = doc.due_date.isoformat()
        if doc.purchase_order is not None:
            fields["purchase_order"] = doc.purchase_order
        fields["vendor.name"] = doc.vendor.name
        if doc.vendor.tax_id is not None:
            fields["vendor.tax_id"] = doc.vendor.tax_id
        if doc.vendor.iban is not None:
            fields["vendor.iban"] = doc.vendor.iban
        if doc.customer is not None:
            fields["customer.name"] = doc.customer.name
        fields["subtotal"] = str(doc.subtotal)
        fields["tax_total"] = str(doc.tax_total)
        fields["total"] = str(doc.total)
    else:
        assert isinstance(doc, Receipt)
        if doc.receipt_number is not None:
            fields["receipt_number"] = doc.receipt_number
        fields["transaction_date"] = doc.transaction_date.isoformat()
        fields["merchant.name"] = doc.merchant.name
        fields["subtotal"] = str(doc.subtotal)
        fields["tax_total"] = str(doc.tax_total)
        fields["tip_total"] = str(doc.tip_total)
        fields["total"] = str(doc.total)

    for idx, item in enumerate(doc.line_items):
        fields[f"line_items[{idx}].amount"] = str(item.amount)

    for idx, tax in enumerate(doc.taxes):
        fields[f"taxes[{idx}].amount"] = str(tax.amount)

    return fields


def _lookup_field_bbox(doc: ExtractedDocument, field_path: str) -> BoundingBox | None:
    if field_path in doc.field_bboxes:
        return doc.field_bboxes[field_path]
    if field_path.startswith("line_items[") and field_path.endswith("].amount"):
        prefix = field_path.split(".")[0]
        idx = int(prefix[len("line_items[") : -1])
        if 0 <= idx < len(doc.line_items):
            return doc.line_items[idx].source_bbox or doc.field_bboxes.get(prefix)
    if field_path.startswith("taxes[") and field_path.endswith("].amount"):
        prefix = field_path.split(".")[0]
        idx = int(prefix[len("taxes[") : -1])
        if 0 <= idx < len(doc.taxes):
            return doc.taxes[idx].source_bbox or doc.field_bboxes.get(prefix)
    if isinstance(doc, Invoice):
        if field_path.startswith("vendor."):
            return doc.vendor.source_bbox
        if field_path.startswith("customer.") and doc.customer is not None:
            return doc.customer.source_bbox
    elif isinstance(doc, Receipt) and field_path.startswith("merchant."):
        return doc.merchant.source_bbox
    return None


def _compute_ocr_score_for_bbox(
    bbox: BoundingBox | None,
    layouts: Sequence[PageLayout],
) -> float:
    """Return mean token confidence for tokens overlapping `bbox` (or page mean if ungrounded)."""
    if not layouts:
        return 1.0

    if bbox is not None:
        for page in layouts:
            if page.page_number != bbox.page:
                continue
            overlapping = [
                tok.confidence
                for tok in page.words
                if tok.bbox.iou(bbox) > 0.0
                or bbox.contains_point(
                    (tok.bbox.x0 + tok.bbox.x1) * 0.5,
                    (tok.bbox.y0 + tok.bbox.y1) * 0.5,
                )
            ]
            if overlapping:
                return round(sum(overlapping) / len(overlapping), 4)

    all_confs = [tok.confidence for page in layouts for tok in page.words]
    if not all_confs:
        return 1.0
    return round(sum(all_confs) / len(all_confs), 4)


def compute_consistency_agreement(
    primary: ExtractedDocument,
    candidates: Sequence[ExtractedDocument] | None = None,
) -> dict[str, float]:
    """Compute per-field agreement fraction across `primary` and additional sampled `candidates`."""
    primary_fields = extract_key_fields(primary)
    if not candidates:
        return {k: 1.0 for k in primary_fields}

    all_docs = [primary, *candidates]
    candidate_maps = [extract_key_fields(d) for d in all_docs]
    n_total = len(candidate_maps)

    agreement: dict[str, float] = {}
    for field_path, target_val in primary_fields.items():
        norm_target = target_val.strip().lower()
        matches = sum(
            1 for cmap in candidate_maps if cmap.get(field_path, "").strip().lower() == norm_target
        )
        agreement[field_path] = round(matches / n_total, 4)
    return agreement


def _compute_rule_scores(
    field_paths: Sequence[str],
    validation: ValidationReport | None,
) -> dict[str, float]:
    scores = {fp: 1.0 for fp in field_paths}
    if validation is None:
        return scores

    for res in validation.results:
        if res.passed or res.field_path is None:
            continue
        fp = res.field_path
        penalty_score = 0.15 if res.severity == Severity.ERROR else 0.55
        if fp in scores:
            scores[fp] = min(scores[fp], penalty_score)
        # Also penalize `tax_total` if a child `taxes[i]` rule failed
        if fp.startswith("taxes[") and "tax_total" in scores:
            scores["tax_total"] = min(scores["tax_total"], penalty_score)

    return scores


def score_document_confidence(
    doc: ExtractedDocument,
    *,
    layouts: Sequence[PageLayout] | None = None,
    validation: ValidationReport | None = None,
    candidates: Sequence[ExtractedDocument] | None = None,
    settings: Settings | None = None,
) -> ConfidenceReport:
    """Fuse OCR, bbox grounding, self-consistency, and validation rules into a `ConfidenceReport`.

    Per-field formula:
        base = 0.30 * ocr_score + 0.30 * grounding_score + 0.40 * consistency_score
        combined_score = round(base * rule_score, 4)

    A document is marked `needs_review=True` if:
    - Any `ERROR`-severity validation rule failed, OR
    - Any field's `combined_score < settings.min_field_confidence`, OR
    - Overall `document_confidence < settings.min_field_confidence`.
    """
    s = settings or get_settings()
    threshold = s.min_field_confidence
    page_layouts = list(layouts) if layouts is not None else []

    key_fields = extract_key_fields(doc)
    consistency_map = compute_consistency_agreement(doc, candidates)
    rule_map = _compute_rule_scores(list(key_fields.keys()), validation)

    field_reports: dict[str, FieldConfidence] = {}
    low_fields: list[str] = []

    for fp, val_str in key_fields.items():
        bbox = _lookup_field_bbox(doc, fp)
        grounding_score = 1.0 if bbox is not None else 0.4
        ocr_score = _compute_ocr_score_for_bbox(bbox, page_layouts)
        consistency_score = consistency_map.get(fp, 1.0)
        rule_score = rule_map.get(fp, 1.0)

        base = (0.30 * ocr_score) + (0.30 * grounding_score) + (0.40 * consistency_score)
        combined = round(max(0.0, min(1.0, base * rule_score)), 4)
        field_needs_review = combined < threshold or rule_score < 0.5

        if field_needs_review:
            low_fields.append(fp)

        field_reports[fp] = FieldConfidence(
            field_path=fp,
            value_str=val_str,
            ocr_score=ocr_score,
            grounding_score=grounding_score,
            consistency_score=consistency_score,
            rule_score=rule_score,
            combined_score=combined,
            needs_review=field_needs_review,
        )

    if field_reports:
        scores_list = [f.combined_score for f in field_reports.values()]
        doc_conf = round(sum(scores_list) / len(scores_list), 4)
        min_conf = round(min(scores_list), 4)
    else:
        doc_conf = 0.0
        min_conf = 0.0

    validation_failed = validation is not None and not validation.passed
    needs_review = validation_failed or bool(low_fields) or (doc_conf < threshold)

    return ConfidenceReport(
        document_confidence=doc_conf,
        min_field_confidence=min_conf,
        needs_review=needs_review,
        low_confidence_fields=low_fields,
        fields=field_reports,
    )
