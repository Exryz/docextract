"""Deterministic IBAN (ISO 13616 mod-97) and tax registration ID format/checksum validators."""

from __future__ import annotations

import re

_IBAN_CLEAN_RE = re.compile(r"[\s\-]+")
_IBAN_SHAPE_RE = re.compile(r"^[A-Z]{2}\d{2}[A-Z0-9]{11,30}$")

# Country-specific IBAN lengths for common jurisdictions
_IBAN_COUNTRY_LENGTHS: dict[str, int] = {
    "AT": 20,
    "BE": 16,
    "CH": 21,
    "DE": 22,
    "DK": 18,
    "ES": 24,
    "FI": 18,
    "FR": 27,
    "GB": 22,
    "IE": 22,
    "IT": 27,
    "LU": 20,
    "NL": 18,
    "NO": 15,
    "PL": 28,
    "PT": 25,
    "SE": 24,
}

# Supported tax ID formats:
# - US EIN: "12-3456789" or "US-12-3456789"
# - PH TIN: "123-456-789" or "123-456-789-000" (optionally prefixed with "PH-")
# - EU VAT: 2-letter country prefix + 8-12 alphanumeric chars (e.g. "DE123456789")
_US_EIN_RE = re.compile(r"^(?:US-)?\d{2}-\d{7}$")
_PH_TIN_RE = re.compile(r"^(?:PH-)?\d{3}-\d{3}-\d{3}(?:-\d{3,4})?$")
_EU_VAT_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{8,12}$")


def validate_iban(raw_iban: str) -> tuple[bool, str | None]:
    """Validate an IBAN using ISO 13616 structure, country length, and mod-97-10 checksum.

    Returns `(True, None)` when valid, or `(False, reason)` when invalid.
    """
    cleaned = _IBAN_CLEAN_RE.sub("", raw_iban).upper()
    if not _IBAN_SHAPE_RE.match(cleaned):
        return False, f"IBAN '{raw_iban}' does not match ISO 13616 alphanumeric format"

    country = cleaned[:2]
    expected_len = _IBAN_COUNTRY_LENGTHS.get(country)
    if expected_len is not None and len(cleaned) != expected_len:
        return (
            False,
            f"IBAN for country '{country}' must be {expected_len} characters (got {len(cleaned)})",
        )

    # Move first 4 chars to end and expand letters A=10 .. Z=35
    rearranged = cleaned[4:] + cleaned[:4]
    numeric_digits = "".join(
        str(ord(ch) - 55) if "A" <= ch <= "Z" else ch for ch in rearranged
    )
    if int(numeric_digits) % 97 != 1:
        return False, f"IBAN '{raw_iban}' failed mod-97 checksum verification"

    return True, None


def validate_tax_id(raw_tax_id: str) -> tuple[bool, str | None]:
    """Validate a vendor/customer tax registration identifier (US EIN, PH TIN, or EU VAT)."""
    cleaned = raw_tax_id.strip().upper()
    if not cleaned:
        return False, "Tax ID is empty"

    if _US_EIN_RE.match(cleaned) or _PH_TIN_RE.match(cleaned) or _EU_VAT_RE.match(cleaned):
        return True, None

    return (
        False,
        f"Tax ID '{raw_tax_id}' does not match US EIN, PH TIN, or EU VAT registration format",
    )
