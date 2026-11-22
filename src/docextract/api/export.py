"""CSV and JSON export router for verified invoice and receipt records."""

from __future__ import annotations

import csv
import io
import json
from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import PlainTextResponse

from docextract.store.models import DocumentRecord
from docextract.store.repository import DocumentRepository

CSV_COLUMNS = (
    "id",
    "status",
    "kind",
    "vendor_name",
    "document_number",
    "issue_or_tx_date",
    "due_date",
    "currency",
    "subtotal",
    "tax_total",
    "total_amount",
    "document_confidence",
    "validation_passed",
    "filename",
    "doc_sha256",
    "created_at",
)


def _record_to_export_row(rec: DocumentRecord) -> dict[str, Any]:
    extracted: dict[str, Any] = json.loads(rec.extracted_json) if rec.extracted_json else {}
    validation: dict[str, Any] = json.loads(rec.validation_json) if rec.validation_json else {}

    issue_or_tx = extracted.get("issue_date") or extracted.get("transaction_date") or ""
    due_date = extracted.get("due_date") or ""
    subtotal = str(extracted.get("subtotal", ""))
    tax_total = str(extracted.get("tax_total", ""))
    val_passed = validation.get("passed")

    return {
        "id": rec.id,
        "status": rec.status,
        "kind": rec.kind or "",
        "vendor_name": rec.vendor_name or "",
        "document_number": rec.document_number or "",
        "issue_or_tx_date": issue_or_tx,
        "due_date": due_date,
        "currency": rec.currency or "",
        "subtotal": subtotal,
        "tax_total": tax_total,
        "total_amount": rec.total_amount or "",
        "document_confidence": (
            f"{rec.document_confidence:.4f}" if rec.document_confidence is not None else ""
        ),
        "validation_passed": "" if val_passed is None else str(bool(val_passed)).lower(),
        "filename": rec.filename,
        "doc_sha256": rec.doc_sha256,
        "created_at": rec.created_at.isoformat(),
    }


def render_csv_export(records: list[DocumentRecord]) -> str:
    """Serialize a list of `DocumentRecord`s into an RFC 4180 CSV string."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(CSV_COLUMNS), lineterminator="\n")
    writer.writeheader()
    for rec in records:
        writer.writerow(_record_to_export_row(rec))
    return buf.getvalue()


def build_export_router(repo: DocumentRepository) -> APIRouter:
    router = APIRouter(tags=["export"])

    @router.get("/export.csv", response_class=PlainTextResponse)
    def export_csv(
        status: str | None = Query(
            default=None,
            description="Filter exported rows by status (e.g. 'approved')",
        ),
        kind: str | None = Query(
            default=None,
            description="Filter exported rows by kind ('invoice' or 'receipt')",
        ),
        limit: int = Query(default=500, ge=1, le=5000),
    ) -> PlainTextResponse:
        records = repo.list_documents(status=status, kind=kind, limit=limit)
        csv_text = render_csv_export(records)
        return PlainTextResponse(
            content=csv_text,
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="docextract_export.csv"'},
        )

    @router.get("/export.json")
    def export_json(
        status: str | None = Query(default=None),
        kind: str | None = Query(default=None),
        limit: int = Query(default=500, ge=1, le=5000),
    ) -> dict[str, Any]:
        records = repo.list_documents(status=status, kind=kind, limit=limit)
        items: list[dict[str, Any]] = []
        for rec in records:
            row = _record_to_export_row(rec)
            row["extracted"] = json.loads(rec.extracted_json) if rec.extracted_json else None
            items.append(row)
        return {"count": len(items), "documents": items}

    return router
