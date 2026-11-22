"""HMAC-SHA256 webhook signing, verification, and exponential-backoff delivery client."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

SIGNATURE_HEADER = "X-Docextract-Signature"
TIMESTAMP_HEADER = "X-Docextract-Timestamp"
EVENT_HEADER = "X-Docextract-Event"


def sign_webhook_payload(
    payload_bytes: bytes,
    secret: str,
    timestamp: int | None = None,
) -> tuple[str, int]:
    """Compute `sha256=<hex>` HMAC signature over `<timestamp>.<payload_bytes>`."""
    ts = timestamp if timestamp is not None else int(time.time())
    signing_input = f"{ts}.".encode() + payload_bytes
    digest = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).hexdigest()
    return f"sha256={digest}", ts


def verify_webhook_signature(
    payload_bytes: bytes,
    secret: str,
    signature_header: str,
    timestamp_header: str | int,
    *,
    tolerance_s: int = 300,
    now_ts: int | None = None,
) -> bool:
    """Verify an HMAC-SHA256 webhook signature in constant time with replay-window protection."""
    try:
        ts = int(timestamp_header)
    except (ValueError, TypeError):
        return False

    ref_now = now_ts if now_ts is not None else int(time.time())
    if abs(ref_now - ts) > tolerance_s:
        return False

    expected_sig, _ = sign_webhook_payload(payload_bytes, secret, timestamp=ts)
    return hmac.compare_digest(expected_sig, signature_header.strip())


@dataclass(frozen=True)
class WebhookDeliveryResult:
    """Outcome of delivering a signed outbound webhook event."""

    delivered: bool
    status_code: int | None
    attempts: int
    last_error: str | None = None


class WebhookSender(Protocol):
    """Protocol for outbound webhook delivery clients."""

    async def send_event(
        self,
        event_type: str,
        payload: dict[str, Any],
    ) -> WebhookDeliveryResult: ...


class HttpWebhookSender:
    """Deliver HMAC-signed JSON webhook events over HTTP with exponential backoff retries."""

    def __init__(
        self,
        url: str,
        secret: str | None = None,
        *,
        max_retries: int = 3,
        base_delay_s: float = 0.25,
        timeout_s: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.url = url
        self.secret = secret
        self.max_retries = max(1, max_retries)
        self.base_delay_s = base_delay_s
        self.timeout_s = timeout_s
        self.transport = transport

    async def send_event(
        self,
        event_type: str,
        payload: dict[str, Any],
    ) -> WebhookDeliveryResult:
        body_bytes = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            EVENT_HEADER: event_type,
        }
        if self.secret:
            sig, ts = sign_webhook_payload(body_bytes, self.secret)
            headers[SIGNATURE_HEADER] = sig
            headers[TIMESTAMP_HEADER] = str(ts)

        last_status: int | None = None
        last_err: str | None = None

        async with httpx.AsyncClient(timeout=self.timeout_s, transport=self.transport) as client:
            for attempt in range(1, self.max_retries + 1):
                try:
                    resp = await client.post(self.url, content=body_bytes, headers=headers)
                    last_status = resp.status_code
                    if 200 <= resp.status_code < 300:
                        return WebhookDeliveryResult(
                            delivered=True,
                            status_code=resp.status_code,
                            attempts=attempt,
                        )
                    last_err = f"HTTP {resp.status_code}"
                    # 4xx client errors (except 429 rate limit) are not retried
                    if 400 <= resp.status_code < 500 and resp.status_code != 429:
                        return WebhookDeliveryResult(
                            delivered=False,
                            status_code=resp.status_code,
                            attempts=attempt,
                            last_error=last_err,
                        )
                except Exception as exc:
                    last_err = str(exc)

                if attempt < self.max_retries and self.base_delay_s > 0:
                    await asyncio.sleep(self.base_delay_s * (2 ** (attempt - 1)))

        return WebhookDeliveryResult(
            delivered=False,
            status_code=last_status,
            attempts=self.max_retries,
            last_error=last_err,
        )
