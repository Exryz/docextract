"""Integration tests for the Jinja2 + HTMX human review queue UI and interactive bbox views."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from datagen import FaultType, LayoutStyle, generate_sample
from docextract.api import create_app
from docextract.config import Settings
from docextract.extract import ScriptedLLM
from docextract.store import DocumentRepository


def test_review_ui_queue_detail_svg_bboxes_and_form_submission(tmp_path: Path) -> None:
    sample = generate_sample(
        seed=801,
        layout=LayoutStyle.CORPORATE_TABLE,
        fault=FaultType.WRONG_TOTAL,
    )
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'ui.db').as_posix()}",
        storage_dir=tmp_path / "storage",
        target_dpi=100,
    )
    repo = DocumentRepository.from_url(settings.database_url)
    llm = ScriptedLLM([sample.injected_document.model_dump_json()])
    app = create_app(repo=repo, llm=llm, settings=settings)
    client = TestClient(app)

    # Upload faulty document so it lands in `needs_review`
    up_resp = client.post(
        "/documents",
        files={"file": ("inv_801_ui.pdf", sample.pdf_bytes, "application/pdf")},
    )
    assert up_resp.status_code == 202
    doc_id = up_resp.json()["document_id"]

    # 1. GET /review queue listing
    queue_html = client.get("/review?status=needs_review")
    assert queue_html.status_code == 200
    assert doc_id in queue_html.text
    assert "needs_review" in queue_html.text

    # 2. GET /review/{doc_id} side-by-side detail page with SVG bounding boxes
    detail_html = client.get(f"/review/{doc_id}")
    assert detail_html.status_code == 200
    assert 'class="bbox-svg"' in detail_html.text
    assert 'data-field="total"' in detail_html.text
    assert 'data-rule-id="grand_total_math"' in detail_html.text

    # 3. POST /review/{doc_id}/submit form with corrected `field__total`
    correct_total = str(sample.gold_document.total)
    submit_resp = client.post(
        f"/review/{doc_id}/submit",
        data={
            "reviewer": "paul.reviewer",
            "action": "approved",
            "field__total": correct_total,
            "note": "Verified against line item sum in UI",
        },
        follow_redirects=True,
    )
    assert submit_resp.status_code == 200
    assert "Status: approved" in submit_resp.text
    assert "paul.reviewer" in submit_resp.text
    assert correct_total in submit_resp.text

    # 4. Missing document returns 404
    assert client.get("/review/nonexistent_doc").status_code == 404
