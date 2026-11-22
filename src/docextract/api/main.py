"""FastAPI REST endpoints for document ingestion, status, page bbox overlays, and review."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import Response
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

from docextract.api.export import build_export_router
from docextract.api.webhooks import HttpWebhookSender, WebhookSender
from docextract.config import Settings, get_settings
from docextract.extract.llm import ExtractionLLM
from docextract.ingest.sniffer import (
    CorruptDocumentError,
    IngestError,
    PageLimitExceededError,
    PayloadTooLargeError,
    UnsupportedMediaTypeError,
)
from docextract.ingest.splitter import ingest_bytes
from docextract.jobs.queue import JobQueue
from docextract.schemas.grounding import BoundingBox
from docextract.store.models import (
    DocumentNotFoundError,
    DocumentRecord,
    DocumentStatus,
    InvalidStateTransitionError,
)
from docextract.store.repository import DocumentRepository
from docextract.ui.router import build_ui_router
from docextract.vision.ocr import OcrEngine


class UploadDocumentResponse(BaseModel):
    document_id: str
    job_id: str
    status: str
    doc_sha256: str
    page_count: int
    is_duplicate_file: bool


class ReviewRequest(BaseModel):
    reviewer: str = Field(min_length=1)
    action: DocumentStatus
    field_overrides: dict[str, Any] | None = None
    note: str | None = None


def _serialize_record(rec: DocumentRecord, job_id: str | None = None) -> dict[str, Any]:
    return {
        "id": rec.id,
        "job_id": job_id,
        "doc_sha256": rec.doc_sha256,
        "filename": rec.filename,
        "mime_type": rec.mime_type,
        "page_count": rec.page_count,
        "status": rec.status,
        "kind": rec.kind,
        "vendor_name": rec.vendor_name,
        "document_number": rec.document_number,
        "currency": rec.currency,
        "total_amount": rec.total_amount,
        "document_confidence": rec.document_confidence,
        "extracted": json.loads(rec.extracted_json) if rec.extracted_json else None,
        "validation": json.loads(rec.validation_json) if rec.validation_json else None,
        "confidence": json.loads(rec.confidence_json) if rec.confidence_json else None,
        "pages": json.loads(rec.pages_json) if rec.pages_json else [],
        "error_message": rec.error_message,
        "created_at": rec.created_at.isoformat(),
        "updated_at": rec.updated_at.isoformat(),
    }


def _render_bbox_overlay(
    image_bytes: bytes,
    field_bboxes: dict[str, BoundingBox],
    page_number: int,
    low_confidence_fields: set[str],
) -> bytes:
    """Draw colored bounding-box rectangles for all grounded fields on `page_number`."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w_px, h_px = img.size

    for field_name, bbox in field_bboxes.items():
        if bbox.page != page_number:
            continue
        left, top, right, bottom = bbox.to_pixel_rect(w_px, h_px)
        # Pad by 2px for visual clarity
        left = max(0, left - 2)
        top = max(0, top - 2)
        right = min(w_px - 1, right + 2)
        bottom = min(h_px - 1, bottom + 2)

        is_low = field_name in low_confidence_fields
        stroke_color = (220, 38, 38, 230) if is_low else (16, 185, 129, 220)
        fill_color = (220, 38, 38, 45) if is_low else (16, 185, 129, 40)
        draw.rectangle((left, top, right, bottom), fill=fill_color, outline=stroke_color, width=3)

    composited = Image.alpha_composite(img, overlay).convert("RGB")
    buf = io.BytesIO()
    composited.save(buf, format="PNG")
    return buf.getvalue()


def create_app(
    *,
    repo: DocumentRepository | None = None,
    queue: JobQueue | None = None,
    llm: ExtractionLLM | None = None,
    ocr_engine: OcrEngine | None = None,
    webhook_sender: WebhookSender | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    s = settings or get_settings()
    actual_repo = repo or DocumentRepository.from_url(s.database_url)
    actual_queue = queue or JobQueue(
        actual_repo,
        llm=llm,
        ocr_engine=ocr_engine,
        settings=s,
    )
    actual_webhook: WebhookSender | None = webhook_sender
    if actual_webhook is None and s.webhook_url:
        actual_webhook = HttpWebhookSender(
            url=s.webhook_url,
            secret=s.webhook_secret,
            max_retries=s.webhook_max_retries,
        )

    app = FastAPI(
        title="docextract Verified Document AI API",
        version="0.1.0",
    )
    app.state.settings = s
    app.state.repo = actual_repo
    app.state.queue = actual_queue
    app.state.webhook_sender = actual_webhook
    app.include_router(build_ui_router(actual_repo))
    app.include_router(build_export_router(actual_repo))

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/documents", response_model=UploadDocumentResponse, status_code=202)
    async def upload_document(
        file: UploadFile = File(...),  # noqa: B008
        run_sync: bool = Query(
            default=True,
            description="Process the extraction job immediately before returning",
        ),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> UploadDocumentResponse:
        raw_bytes = await file.read()
        try:
            ingest_res, _ = ingest_bytes(
                raw_bytes,
                settings=s,
                storage_dir=s.storage_dir,
                persist=True,
            )
        except PayloadTooLargeError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except UnsupportedMediaTypeError as exc:
            raise HTTPException(status_code=415, detail=str(exc)) from exc
        except (PageLimitExceededError, CorruptDocumentError, IngestError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        # If idempotency key already exists, return its existing document and job
        if idempotency_key:
            existing_job = actual_queue.enqueue(
                document_id="pending",
                idempotency_key=idempotency_key,
            )
            if existing_job.document_id != "pending":
                existing_rec = actual_repo.get(existing_job.document_id)
                return UploadDocumentResponse(
                    document_id=existing_rec.id,
                    job_id=existing_job.id,
                    status=existing_rec.status,
                    doc_sha256=existing_rec.doc_sha256,
                    page_count=existing_rec.page_count,
                    is_duplicate_file=True,
                )

        rec = actual_repo.create_from_ingest(
            ingest_res,
            filename=file.filename or "upload.pdf",
        )
        job = actual_queue.enqueue(rec.id, idempotency_key=idempotency_key)
        if job.document_id == "pending":
            # Update placeholder job document_id if created in idempotency pre-check
            from sqlmodel import Session

            with Session(actual_repo.engine) as session:
                db_job = session.get(type(job), job.id)
                if db_job is not None:
                    db_job.document_id = rec.id
                    session.add(db_job)
                    session.commit()

        if run_sync:
            await actual_queue.process_job(job.id)
            rec = actual_repo.get(rec.id)
            if actual_webhook is not None:
                await actual_webhook.send_event(
                    f"document.{rec.status}",
                    _serialize_record(rec, job_id=job.id),
                )

        return UploadDocumentResponse(
            document_id=rec.id,
            job_id=job.id,
            status=rec.status,
            doc_sha256=rec.doc_sha256,
            page_count=rec.page_count,
            is_duplicate_file=ingest_res.is_duplicate,
        )

    @app.get("/documents")
    def list_documents(
        status: str | None = Query(default=None),
        kind: str | None = Query(default=None),
        limit: int = Query(default=50, ge=1, le=200),
    ) -> list[dict[str, Any]]:
        records = actual_repo.list_documents(status=status, kind=kind, limit=limit)
        return [_serialize_record(r) for r in records]

    @app.get("/documents/{doc_id}")
    def get_document(doc_id: str) -> dict[str, Any]:
        try:
            rec = actual_repo.get(doc_id)
        except DocumentNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        job = actual_queue.get_job_for_document(doc_id)
        payload = _serialize_record(rec, job_id=job.id if job else None)
        payload["audit_trail"] = [
            {
                "id": a.id,
                "reviewer": a.reviewer,
                "action": a.action,
                "field_path": a.field_path,
                "before_value": a.before_value,
                "after_value": a.after_value,
                "note": a.note,
                "created_at": a.created_at.isoformat(),
            }
            for a in actual_repo.list_audit_entries(doc_id)
        ]
        return payload

    @app.get("/documents/{doc_id}/pages/{page_number}.png")
    def get_page_image(
        doc_id: str,
        page_number: int,
        overlay: bool = Query(default=True, description="Render field bounding-box overlays"),
    ) -> Response:
        try:
            rec = actual_repo.get(doc_id)
        except DocumentNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        pages_meta: list[dict[str, Any]] = json.loads(rec.pages_json) if rec.pages_json else []
        page_entry = next((p for p in pages_meta if p.get("page_number") == page_number), None)
        if page_entry is None or not page_entry.get("image_path"):
            raise HTTPException(
                status_code=404,
                detail=f"Page {page_number} not found for document '{doc_id}'",
            )

        img_path = Path(str(page_entry["image_path"]))
        if not img_path.exists():
            raise HTTPException(status_code=404, detail="Raster page file missing on disk")

        raw_png = img_path.read_bytes()
        if not overlay or not rec.extracted_json:
            return Response(content=raw_png, media_type="image/png")

        doc = actual_repo.get_extracted_document(doc_id)
        if doc is None or not doc.field_bboxes:
            return Response(content=raw_png, media_type="image/png")

        low_fields: set[str] = set()
        if rec.confidence_json:
            conf_data = json.loads(rec.confidence_json)
            low_fields = set(conf_data.get("low_confidence_fields", []))

        rendered_png = _render_bbox_overlay(raw_png, doc.field_bboxes, page_number, low_fields)
        return Response(content=rendered_png, media_type="image/png")

    @app.post("/documents/{doc_id}/review")
    async def review_document(doc_id: str, body: ReviewRequest) -> dict[str, Any]:
        try:
            updated = actual_repo.apply_review(
                doc_id,
                reviewer=body.reviewer,
                action=body.action,
                field_overrides=body.field_overrides,
                note=body.note,
            )
        except DocumentNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (InvalidStateTransitionError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        doc_payload = get_document(updated.id)
        if actual_webhook is not None:
            await actual_webhook.send_event(f"document.{updated.status}", doc_payload)
        return doc_payload

    @app.get("/jobs/{job_id}")
    def get_job_status(job_id: str) -> dict[str, Any]:
        job = actual_queue.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
        return {
            "id": job.id,
            "document_id": job.document_id,
            "idempotency_key": job.idempotency_key,
            "status": job.status,
            "attempts": job.attempts,
            "max_attempts": job.max_attempts,
            "last_error": job.last_error,
        }

    return app


def app_factory() -> FastAPI:
    return create_app()
