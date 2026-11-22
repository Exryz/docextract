"""Unit and integration tests for HMAC-signed outbound webhooks and CSV/JSON exports."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from datagen import FaultType, LayoutStyle, generate_sample
from docextract.api import (
    EVENT_HEADER,
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    HttpWebhookSender,
    WebhookDeliveryResult,
    create_app,
    sign_webhook_payload,
    verify_webhook_signature,
)
from docextract.config import Settings
from docextract.extract import ScriptedLLM
from docextract.store import DocumentRepository


def test_webhook_hmac_signing_verification_and_replay_window() -> None:
    payload = b'{"document_id":"doc_123","status":"approved"}'
    secret = "whsec_northwind_test_secret_key"
    sig, ts = sign_webhook_payload(payload, secret, timestamp=1700000000)

    assert sig.startswith("sha256=")
    assert verify_webhook_signature(
        payload,
        secret,
        sig,
        ts,
        tolerance_s=300,
        now_ts=1700000100,
    )

    # Tampered payload fails verification
    assert not verify_webhook_signature(
        b'{"document_id":"doc_123","status":"rejected"}',
        secret,
        sig,
        ts,
        now_ts=1700000100,
    )

    # Stale timestamp outside 300s replay window fails verification
    assert not verify_webhook_signature(
        payload,
        secret,
        sig,
        ts,
        tolerance_s=300,
        now_ts=1700000500,
    )


@pytest.mark.asyncio
async def test_http_webhook_sender_exponential_backoff_retries() -> None:
    attempts = 0
    received_headers: dict[str, str] = {}
    received_body = b""

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts, received_headers, received_body
        attempts += 1
        received_headers = dict(request.headers)
        received_body = request.content
        if attempts < 3:
            return httpx.Response(503, json={"error": "temporary outage"})
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(mock_handler)
    sender = HttpWebhookSender(
        url="https://hooks.northwind.example/docextract",
        secret="whsec_retry_key",
        max_retries=3,
        base_delay_s=0.001,
        transport=transport,
    )

    res = await sender.send_event("document.approved", {"id": "doc_99", "total": "450.00"})
    assert res.delivered is True
    assert res.status_code == 200
    assert res.attempts == 3
    assert received_headers.get(EVENT_HEADER.lower()) == "document.approved"
    assert verify_webhook_signature(
        received_body,
        "whsec_retry_key",
        received_headers[SIGNATURE_HEADER.lower()],
        received_headers[TIMESTAMP_HEADER.lower()],
    )


@pytest.mark.asyncio
async def test_http_webhook_sender_does_not_retry_4xx_client_error() -> None:
    calls = 0

    async def mock_400(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, json={"error": "bad request"})

    sender = HttpWebhookSender(
        url="https://hooks.northwind.example/docextract",
        max_retries=3,
        base_delay_s=0.001,
        transport=httpx.MockTransport(mock_400),
    )
    res = await sender.send_event("document.rejected", {"id": "doc_bad"})
    assert res.delivered is False
    assert res.status_code == 400
    assert res.attempts == 1
    assert calls == 1


class _RecordingWebhookSender:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def send_event(
        self,
        event_type: str,
        payload: dict[str, Any],
    ) -> WebhookDeliveryResult:
        self.events.append((event_type, payload))
        return WebhookDeliveryResult(delivered=True, status_code=200, attempts=1)


def test_export_csv_and_json_and_webhook_dispatch_on_upload_and_review(tmp_path: Path) -> None:
    clean_sample = generate_sample(
        seed=901, layout=LayoutStyle.CORPORATE_TABLE, fault=FaultType.NONE
    )
    faulty_sample = generate_sample(
        seed=902, layout=LayoutStyle.MODERN_SIDEBAR, fault=FaultType.WRONG_TOTAL
    )

    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'export.db').as_posix()}",
        storage_dir=tmp_path / "storage",
        target_dpi=100,
    )
    repo = DocumentRepository.from_url(settings.database_url)
    llm = ScriptedLLM(
        [
            clean_sample.gold_document.model_dump_json(),
            faulty_sample.injected_document.model_dump_json(),
        ]
    )
    recorder = _RecordingWebhookSender()
    app = create_app(repo=repo, llm=llm, webhook_sender=recorder, settings=settings)
    client = TestClient(app)

    # 1. Upload clean invoice -> fires `document.approved` webhook
    r1 = client.post(
        "/documents",
        files={"file": ("clean_901.pdf", clean_sample.pdf_bytes, "application/pdf")},
    )
    assert r1.status_code == 202
    assert recorder.events[-1][0] == "document.approved"

    # 2. Upload faulty invoice -> fires `document.needs_review` webhook
    r2 = client.post(
        "/documents",
        files={"file": ("faulty_902.pdf", faulty_sample.pdf_bytes, "application/pdf")},
    )
    assert r2.status_code == 202
    doc2_id = r2.json()["document_id"]
    assert recorder.events[-1][0] == "document.needs_review"

    # 3. Review faulty invoice -> fires `document.approved` webhook
    rev = client.post(
        f"/documents/{doc2_id}/review",
        json={
            "reviewer": "ops.lead",
            "action": "approved",
            "field_overrides": {"total": str(faulty_sample.gold_document.total)},
        },
    )
    assert rev.status_code == 200
    assert len(recorder.events) == 3
    assert recorder.events[-1][0] == "document.approved"

    # 4. GET /export.csv?status=approved
    csv_resp = client.get("/export.csv?status=approved")
    assert csv_resp.status_code == 200
    assert "text/csv" in csv_resp.headers["content-type"]
    rows = list(csv.DictReader(io.StringIO(csv_resp.text)))
    assert len(rows) == 2
    assert all(r["status"] == "approved" for r in rows)
    assert all(r["validation_passed"] == "true" for r in rows)

    # 5. GET /export.json?status=approved
    json_resp = client.get("/export.json?status=approved")
    assert json_resp.status_code == 200
    json_payload = json_resp.json()
    assert json_payload["count"] == 2
    assert len(json_payload["documents"]) == 2
    assert json_payload["documents"][0]["extracted"] is not None
