"""Public exports for `docextract.scoring`."""

from docextract.scoring.confidence import (
    ConfidenceReport,
    FieldConfidence,
    compute_consistency_agreement,
    extract_key_fields,
    score_document_confidence,
)

__all__ = [
    "ConfidenceReport",
    "FieldConfidence",
    "compute_consistency_agreement",
    "extract_key_fields",
    "score_document_confidence",
]
