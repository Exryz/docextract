"""Jinja2 + HTMX human review queue router with interactive SVG bounding-box overlays."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from docextract.scoring.confidence import extract_key_fields
from docextract.store.models import (
    DocumentNotFoundError,
    DocumentStatus,
    InvalidStateTransitionError,
)
from docextract.store.repository import DocumentRepository

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def build_ui_router(repo: DocumentRepository) -> APIRouter:
    router = APIRouter(tags=["ui"])
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    @router.get("/review", response_class=HTMLResponse)
    def review_queue_page(
        request: Request,
        status: str | None = Query(default=None),
    ) -> HTMLResponse:
        docs = repo.list_documents(status=status, limit=100)
        return templates.TemplateResponse(
            request=request,
            name="queue.html",
            context={
                "documents": docs,
                "active_status": status,
            },
        )

    @router.get("/review/{doc_id}", response_class=HTMLResponse)
    def review_detail_page(
        request: Request,
        doc_id: str,
        page: int = Query(default=1, ge=1),
    ) -> HTMLResponse:
        try:
            rec = repo.get(doc_id)
        except DocumentNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        doc = repo.get_extracted_document(doc_id)
        conf_data: dict[str, Any] = (
            json.loads(rec.confidence_json) if rec.confidence_json else {}
        )
        val_data: dict[str, Any] = (
            json.loads(rec.validation_json) if rec.validation_json else {}
        )
        field_confs: dict[str, Any] = conf_data.get("fields", {})
        low_set = set(conf_data.get("low_confidence_fields", []))

        editable_fields: list[dict[str, Any]] = []
        bboxes: list[dict[str, Any]] = []

        if doc is not None:
            key_fields = extract_key_fields(doc)
            for path, val_str in key_fields.items():
                fc = field_confs.get(path, {})
                score = float(fc.get("combined_score", 1.0))
                needs_rev = bool(fc.get("needs_review", path in low_set))
                editable_fields.append(
                    {
                        "path": path,
                        "value": val_str,
                        "score": score,
                        "needs_review": needs_rev,
                    }
                )

            for field_path, box in doc.field_bboxes.items():
                if box.page != page:
                    continue
                fc = field_confs.get(field_path, {})
                score = float(fc.get("combined_score", 1.0))
                bboxes.append(
                    {
                        "field_path": field_path,
                        "x0": box.x0,
                        "y0": box.y0,
                        "width": box.width,
                        "height": box.height,
                        "is_low": field_path in low_set,
                        "confidence_pct": f"{score * 100:.0f}%",
                    }
                )

        return templates.TemplateResponse(
            request=request,
            name="detail.html",
            context={
                "record": rec,
                "current_page": page,
                "editable_fields": editable_fields,
                "bboxes": bboxes,
                "validation_results": val_data.get("results", []),
                "audit_entries": repo.list_audit_entries(doc_id),
            },
        )

    @router.post("/review/{doc_id}/submit", response_model=None)
    async def submit_review_form(request: Request, doc_id: str) -> RedirectResponse:
        form = await request.form()
        reviewer = str(form.get("reviewer") or "reviewer.ops").strip()
        action_raw = str(form.get("action") or "approved").strip()
        note_raw = str(form.get("note") or "").strip() or None

        try:
            target_status = DocumentStatus(action_raw)
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail=f"Invalid action: {action_raw}"
            ) from exc

        overrides: dict[str, Any] = {}
        for key, value in form.multi_items():
            if key.startswith("field__") and isinstance(value, str):
                field_path = key.removeprefix("field__")
                overrides[field_path] = value.strip()

        try:
            repo.apply_review(
                doc_id,
                reviewer=reviewer,
                action=target_status,
                field_overrides=overrides or None,
                note=note_raw,
            )
        except DocumentNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (InvalidStateTransitionError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return RedirectResponse(url=f"/review/{doc_id}", status_code=303)

    return router
