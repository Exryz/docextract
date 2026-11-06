"""Magic-byte MIME sniffing, SHA-256 content hashing, and upload limit enforcement."""

from __future__ import annotations

import hashlib
from enum import StrEnum


class SupportedMime(StrEnum):
    PDF = "application/pdf"
    PNG = "image/png"
    JPEG = "image/jpeg"
    WEBP = "image/webp"
    TIFF = "image/tiff"


class IngestError(ValueError):
    """Base error raised when an uploaded document fails ingest validation."""


class UnsupportedMediaTypeError(IngestError):
    """Raised when file magic bytes do not match any supported document/image format."""


class PayloadTooLargeError(IngestError):
    """Raised when an uploaded payload exceeds `max_upload_bytes`."""


class PageLimitExceededError(IngestError):
    """Raised when a PDF contains more pages than `max_pages`."""


class CorruptDocumentError(IngestError):
    """Raised when a PDF or image cannot be decoded or has zero pages."""


def sniff_mime(data: bytes) -> SupportedMime:
    """Inspect magic bytes at the start of `data` to determine its MIME type.

    Never trusts file extensions or client-supplied `Content-Type` headers.
    """
    if len(data) < 4:
        raise UnsupportedMediaTypeError("File is too small (< 4 bytes) to contain a valid header")

    # PDF: "%PDF-" anywhere in the first 1024 bytes (per PDF spec, usually offset 0)
    header_window = data[:1024]
    if b"%PDF-" in header_window:
        return SupportedMime.PDF

    # PNG: 89 50 4E 47 0D 0A 1A 0A
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return SupportedMime.PNG

    # JPEG: FF D8 FF
    if data.startswith(b"\xff\xd8\xff"):
        return SupportedMime.JPEG

    # WEBP: "RIFF" + 4-byte size + "WEBP"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return SupportedMime.WEBP

    # TIFF: little-endian "II*\x00" or big-endian "MM\x00*"
    if data.startswith((b"II*\x00", b"MM\x00*")):
        return SupportedMime.TIFF

    raise UnsupportedMediaTypeError(
        "Unsupported file format: magic bytes did not match PDF, PNG, JPEG, WEBP, or TIFF"
    )


def compute_sha256(data: bytes) -> str:
    """Return lowercase hex SHA-256 digest of `data`."""
    if not data:
        raise IngestError("Cannot hash empty payload")
    return hashlib.sha256(data).hexdigest()


def validate_upload_size(data: bytes, max_bytes: int) -> None:
    """Verify payload is non-empty and within `max_bytes`."""
    if not data:
        raise IngestError("Uploaded payload is empty (0 bytes)")
    if len(data) > max_bytes:
        raise PayloadTooLargeError(
            f"Payload size ({len(data):,} bytes) exceeds limit of {max_bytes:,} bytes"
        )
