"""Integration tests for the background job queue/worker and FastAPI REST endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from datagen import FaultType, LayoutStyle, generate_sample
from docextract.api import create_app
from docextract.config import Settings
from docextract.extract import ScriptedLLM
from docextract.store import DocumentRepository, DocumentStatus


def test_api_upload_auto_approve_idempotency_and_page_overlay(tmp_path: Path) -> None:
    sample = generate_sample(seed=701, layout=LayoutStyle.CORPORATE_TABLE, fault=FaultType.NONE)
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
        storage_dir=tmp_path / "storage",
        target_dpi=120,
    )
    repo = DocumentRepository.from_url(settings.database_url)
    llm = ScriptedLLM([sample.gold_document.model_dump_json()])
    app = create_app(repo=repo, llm=llm, settings=settings)
    client = TestClient(app)

    assert client.get("/health").json() == {"status": "ok"}

    # Upload clean invoice with Idempotency-Key
    resp = client.post(
        "/documents",
        files={"file": ("inv_701.pdf", sample.pdf_bytes, "application/pdf")},
        headers={"Idempotency-Key": "idem-key-701"},
    )
    assert resp.status_code == 202
    data = resp.json()
    doc_id = data["document_id"]
    job_id = data["job_id"]
    assert data["status"] == DocumentStatus.APPROVED.value
    assert data["page_count"] == 1

    # Re-sending with identical Idempotency-Key returns same doc & job without calling LLM again
    resp_dup = client.post(
        "/documents",
        files={"file": ("inv_701.pdf", sample.pdf_bytes, "application/pdf")},
        headers={"Idempotency-Key": "idem-key-701"},
    )
    assert resp_dup.status_code == 202
    assert resp_dup.json()["document_id"] == doc_id
    assert resp_dup.json()["job_id"] == job_id
    assert len(llm.calls) == 1

    # Inspect job endpoint
    job_resp = client.get(f"/jobs/{job_id}")
    assert job_resp.status_code == 200
    assert job_resp.json()["status"] == "succeeded"

    # Inspect full document details
    doc_resp = client.get(f"/documents/{doc_id}")
    assert doc_resp.status_code == 200
    doc_payload = doc_resp.json()
    assert doc_payload["status"] == "approved"
    assert doc_payload["validation"]["passed"] is True

    # Fetch rendered PNG page with bounding-box overlay
    png_overlay = client.get(f"/documents/{doc_id}/pages/1.png?overlay=true")
    assert png_overlay.status_code == 200
    assert png_overlay.headers["content-type"] == "image/png"
    assert png_overlay.content.startswith(b"\x89PNG\r\n\x1a\n")

    png_raw = client.get(f"/documents/{doc_id}/pages/1.png?overlay=false")
    assert png_raw.status_code == 200
    # Overlay image differs from raw raster because bbox rectangles were drawn
    assert png_overlay.content != png_raw.content


def test_api_faulty_document_routes_to_review_and_accepts_corrections(tmp_path: Path) -> None:
    sample = generate_sample(
        seed=702,
        layout=LayoutStyle.CORPORATE_TABLE,
        fault=FaultType.WRONG_TOTAL,
    )
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'test2.db').as_posix()}",
        storage_dir=tmp_path / "storage",
        target_dpi=100,
    )
    repo = DocumentRepository.from_url(settings.database_url)
    llm = ScriptedLLM([sample.injected_document.model_dump_json()])
    app = create_app(repo=repo, llm=llm, settings=settings)
    client = TestClient(app)

    resp = client.post(
        "/documents",
        files={"file": ("inv_702_bad.pdf", sample.pdf_bytes, "application/pdf")},
    )
    assert resp.status_code == 202
    doc_id = resp.json()["document_id"]
    assert resp.json()["status"] == DocumentStatus.NEEDS_REVIEW.value

    # List documents filtered by needs_review
    queue_resp = client.get("/documents?status=needs_review")
    assert queue_resp.status_code == 200
    assert len(queue_resp.json()) == 1

    # Submit human review fixing `total` and approving
    review_resp = client.post(
        f"/documents/{doc_id}/review",
        json={
            "reviewer": "reviewer.jane",
            "action": "approved",
            "field_overrides": {"total": str(sample.gold_document.total)},
            "note": "Fixed total calculation",
        },
    )
    assert review_resp.status_code == 200
    reviewed_data = review_resp.json()
    assert reviewed_data["status"] == "approved"
    assert reviewed_data["total_amount"] == str(sample.gold_document.total)
    assert reviewed_data["validation"]["passed"] is True
    assert len(reviewed_data["audit_trail"]) == 2


def test_api_rejects_unsupported_mime_and_detects_duplicate_vendor_invoice(
    tmp_path: Path,
) -> None:
    sample = generate_sample(seed=703, layout=LayoutStyle.CORPORATE_TABLE, fault=FaultType.NONE)
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'test3.db').as_posix()}",
        storage_dir=tmp_path / "storage",
        target_dpi=100,
    )
    repo = DocumentRepository.from_url(settings.database_url)
    llm = ScriptedLLM(
        [
            sample.gold_document.model_dump_json(),
            sample.gold_document.model_dump_json(),
        ]
    )
    app = create_app(repo=repo, llm=llm, settings=settings)
    client = TestClient(app)

    # Unsupported file format returns 415
    bad_resp = client.post(
        "/documents",
        files={"file": ("exploit.exe", b"MZ\x90\x00\x03\x00\x00\x00", "application/octet-stream")},
    )
    assert bad_resp.status_code == 415

    # First upload of invoice 703 auto-approves
    r1 = client.post(
        "/documents",
        files={"file": ("inv_703_a.pdf", sample.pdf_bytes, "application/pdf")},
    )
    assert r1.json()["status"] == "approved"

    # Second upload with same vendor + invoice_number is flagged by `duplicate_vendor_invoice` rule
    r2 = client.post(
        "/documents",
        files={"file": ("inv_703_b.pdf", sample.pdf_bytes, "application/pdf")},
    )
    assert r2.json()["status"] == "needs_review"
    doc2 = client.get(f"/documents/{r2.json()['document_id']}").json()
    failed_rules = [
        r["rule_id"] for r in doc2["validation"]["results"] if not r["passed"]
    ]
    assert "duplicate_vendor_invoice" in failed_rules
