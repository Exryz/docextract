"""Pluggable validation rule registry and deterministic math, date, tax, and checksum rules."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date, timedelta
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from docextract.config import Settings, get_settings
from docextract.schemas.document import ExtractedDocument, Invoice, Receipt
from docextract.validate.checksums import validate_iban, validate_tax_id


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class RuleResult(BaseModel):
    """Outcome of executing a single validation rule against an extracted document."""

    model_config = ConfigDict(frozen=True)

    rule_id: str
    passed: bool
    severity: Severity = Severity.ERROR
    field_path: str | None = None
    expected: str | None = None
    actual: str | None = None
    message: str


class ValidationReport(BaseModel):
    """Aggregated validation report across all registered rules for a document."""

    model_config = ConfigDict(frozen=True)

    passed: bool = Field(description="True when zero ERROR-severity rules failed")
    error_count: int = Field(ge=0)
    warning_count: int = Field(ge=0)
    results: list[RuleResult] = Field(default_factory=list)

    @property
    def failed_errors(self) -> list[RuleResult]:
        return [r for r in self.results if not r.passed and r.severity == Severity.ERROR]

    @property
    def failed_warnings(self) -> list[RuleResult]:
        return [r for r in self.results if not r.passed and r.severity == Severity.WARNING]

    @property
    def failed_rule_ids(self) -> list[str]:
        return [r.rule_id for r in self.results if not r.passed]


ValidationRuleFn = Callable[[ExtractedDocument, Settings], Sequence[RuleResult]]


class RuleRegistry:
    """Ordered registry of deterministic validation rules."""

    def __init__(self) -> None:
        self._rules: list[tuple[str, ValidationRuleFn]] = []

    def register(self, rule_id: str, fn: ValidationRuleFn) -> None:
        self._rules.append((rule_id, fn))

    def run(
        self,
        doc: ExtractedDocument,
        settings: Settings | None = None,
    ) -> ValidationReport:
        s = settings or get_settings()
        all_results: list[RuleResult] = []
        for _, rule_fn in self._rules:
            all_results.extend(rule_fn(doc, s))

        errors = sum(1 for r in all_results if not r.passed and r.severity == Severity.ERROR)
        warnings = sum(1 for r in all_results if not r.passed and r.severity == Severity.WARNING)
        return ValidationReport(
            passed=(errors == 0),
            error_count=errors,
            warning_count=warnings,
            results=all_results,
        )


def check_line_item_amounts(doc: ExtractedDocument, settings: Settings) -> list[RuleResult]:
    """Verify `(quantity * unit_price) - discount == amount` for every line item."""
    tol = settings.rounding_tolerance
    results: list[RuleResult] = []
    for idx, item in enumerate(doc.line_items):
        expected = item.expected_amount(doc.currency)
        diff = abs(item.amount - expected)
        passed = diff <= tol
        results.append(
            RuleResult(
                rule_id="line_item_math",
                passed=passed,
                severity=Severity.ERROR,
                field_path=f"line_items[{idx}].amount",
                expected=str(expected),
                actual=str(item.amount),
                message=(
                    f"Line item #{idx + 1} amount matches quantity * unit_price"
                    if passed
                    else (
                        f"Line item #{idx + 1} ('{item.description}') amount {item.amount} "
                        f"does not match expected {expected} (diff={diff})"
                    )
                ),
            )
        )
    return results


def check_subtotal_matches_lines(doc: ExtractedDocument, settings: Settings) -> list[RuleResult]:
    """Verify `sum(line_items.amount) == subtotal` within rounding tolerance."""
    tol = settings.rounding_tolerance
    expected = doc.computed_subtotal
    diff = abs(doc.subtotal - expected)
    passed = diff <= tol
    return [
        RuleResult(
            rule_id="subtotal_math",
            passed=passed,
            severity=Severity.ERROR,
            field_path="subtotal",
            expected=str(expected),
            actual=str(doc.subtotal),
            message=(
                "Subtotal matches sum of line items"
                if passed
                else f"Subtotal {doc.subtotal} does not equal sum of line items {expected}"
            ),
        )
    ]


def check_tax_breakdown(doc: ExtractedDocument, settings: Settings) -> list[RuleResult]:
    """Verify tax lines sum to `tax_total`, `rate * base` matches, and tax rate is plausible."""
    tol = settings.rounding_tolerance
    results: list[RuleResult] = []

    # 1. Each explicit TaxLine rate * base check + plausibility (<= 35%)
    for idx, tax in enumerate(doc.taxes):
        if tax.rate is not None and tax.rate > Decimal("0.35"):
            results.append(
                RuleResult(
                    rule_id="tax_rate_plausibility",
                    passed=False,
                    severity=Severity.WARNING,
                    field_path=f"taxes[{idx}].rate",
                    expected="<= 0.35",
                    actual=str(tax.rate),
                    message=f"Tax rate {tax.rate} exceeds 35% plausibility threshold",
                )
            )
        expected_tax = tax.expected_amount(doc.subtotal - doc.discount_total, doc.currency)
        if expected_tax is not None:
            diff = abs(tax.amount - expected_tax)
            passed = diff <= tol
            results.append(
                RuleResult(
                    rule_id="tax_line_math",
                    passed=passed,
                    severity=Severity.ERROR,
                    field_path=f"taxes[{idx}].amount",
                    expected=str(expected_tax),
                    actual=str(tax.amount),
                    message=(
                        f"Tax line '{tax.label}' matches rate * base"
                        if passed
                        else (
                            f"Tax line '{tax.label}' amount {tax.amount} "
                            f"does not match expected {expected_tax}"
                        )
                    ),
                )
            )

    # 2. If tax lines exist, their sum must equal `tax_total`
    if doc.taxes:
        sum_taxes = sum((t.amount for t in doc.taxes), Decimal("0.00"))
        diff_total = abs(doc.tax_total - sum_taxes)
        passed_total = diff_total <= tol
        results.append(
            RuleResult(
                rule_id="tax_total_math",
                passed=passed_total,
                severity=Severity.ERROR,
                field_path="tax_total",
                expected=str(sum_taxes),
                actual=str(doc.tax_total),
                message=(
                    "tax_total matches sum of tax lines"
                    if passed_total
                    else f"tax_total {doc.tax_total} does not match sum of tax lines {sum_taxes}"
                ),
            )
        )
    elif doc.subtotal > Decimal("0") and doc.tax_total > doc.subtotal * Decimal("0.35"):
        results.append(
            RuleResult(
                rule_id="tax_rate_plausibility",
                passed=False,
                severity=Severity.WARNING,
                field_path="tax_total",
                expected=f"<= {doc.subtotal * Decimal('0.35'):.2f}",
                actual=str(doc.tax_total),
                message=f"tax_total {doc.tax_total} exceeds 35% of subtotal {doc.subtotal}",
            )
        )

    return results


def check_grand_total(doc: ExtractedDocument, settings: Settings) -> list[RuleResult]:
    """Verify grand `total` equals `subtotal - discount + tax + shipping/tip`."""
    tol = settings.rounding_tolerance
    expected = doc.computed_total
    diff = abs(doc.total - expected)
    passed = diff <= tol
    return [
        RuleResult(
            rule_id="grand_total_math",
            passed=passed,
            severity=Severity.ERROR,
            field_path="total",
            expected=str(expected),
            actual=str(doc.total),
            message=(
                "Grand total matches subtotal, discounts, taxes, and adjustments"
                if passed
                else f"Grand total {doc.total} does not match expected {expected} (diff={diff})"
            ),
        )
    ]


def check_date_sanity(
    doc: ExtractedDocument,
    settings: Settings,
    reference_today: date | None = None,
) -> list[RuleResult]:
    """Verify `issue_date <= due_date` and dates fall within a plausible calendar window."""
    today = reference_today or date(2026, 12, 31)
    max_future = today + timedelta(days=7)
    min_past = date(2000, 1, 1)
    results: list[RuleResult] = []

    if isinstance(doc, Invoice):
        if doc.issue_date < min_past or doc.issue_date > max_future:
            results.append(
                RuleResult(
                    rule_id="date_range_sanity",
                    passed=False,
                    severity=Severity.WARNING,
                    field_path="issue_date",
                    expected=f"between {min_past.isoformat()} and {max_future.isoformat()}",
                    actual=doc.issue_date.isoformat(),
                    message=(
                        f"Invoice issue_date {doc.issue_date.isoformat()} "
                        "is outside plausible window"
                    ),
                )
            )
        if doc.due_date is not None:
            passed_order = doc.due_date >= doc.issue_date
            results.append(
                RuleResult(
                    rule_id="invoice_due_date_order",
                    passed=passed_order,
                    severity=Severity.ERROR,
                    field_path="due_date",
                    expected=f">= {doc.issue_date.isoformat()}",
                    actual=doc.due_date.isoformat(),
                    message=(
                        "Invoice due_date is on or after issue_date"
                        if passed_order
                        else (
                            f"Invoice due_date ({doc.due_date.isoformat()}) precedes "
                            f"issue_date ({doc.issue_date.isoformat()})"
                        )
                    ),
                )
            )
    else:
        assert isinstance(doc, Receipt)
        passed_range = min_past <= doc.transaction_date <= max_future
        results.append(
            RuleResult(
                rule_id="date_range_sanity",
                passed=passed_range,
                severity=Severity.WARNING,
                field_path="transaction_date",
                expected=f"between {min_past.isoformat()} and {max_future.isoformat()}",
                actual=doc.transaction_date.isoformat(),
                message=(
                    "Receipt transaction_date is within plausible window"
                    if passed_range
                    else (
                        f"Receipt transaction_date {doc.transaction_date.isoformat()} "
                        "is out of range"
                    )
                ),
            )
        )
    return results


def check_party_identifiers(doc: ExtractedDocument, settings: Settings) -> list[RuleResult]:
    """Validate IBAN mod-97 checksums and tax registration IDs on vendor/merchant/customer."""
    results: list[RuleResult] = []
    parties = (
        [("vendor", doc.vendor), *( [("customer", doc.customer)] if doc.customer else [] )]
        if isinstance(doc, Invoice)
        else [("merchant", doc.merchant)]
    )

    for prefix, party in parties:
        if party.iban:
            ok_iban, err_iban = validate_iban(party.iban)
            results.append(
                RuleResult(
                    rule_id="iban_checksum",
                    passed=ok_iban,
                    severity=Severity.ERROR,
                    field_path=f"{prefix}.iban",
                    expected="valid ISO 13616 IBAN",
                    actual=party.iban,
                    message=err_iban or f"Valid IBAN on {prefix}",
                )
            )
        if party.tax_id:
            ok_tax, err_tax = validate_tax_id(party.tax_id)
            results.append(
                RuleResult(
                    rule_id="tax_id_format",
                    passed=ok_tax,
                    severity=Severity.WARNING,
                    field_path=f"{prefix}.tax_id",
                    expected="US EIN / PH TIN / EU VAT",
                    actual=party.tax_id,
                    message=err_tax or f"Valid tax ID format on {prefix}",
                )
            )
    return results


def build_default_registry() -> RuleRegistry:
    """Create a `RuleRegistry` populated with all standard financial and identifier rules."""
    reg = RuleRegistry()
    reg.register("line_item_math", check_line_item_amounts)
    reg.register("subtotal_math", check_subtotal_matches_lines)
    reg.register("tax_breakdown", check_tax_breakdown)
    reg.register("grand_total_math", check_grand_total)
    reg.register("date_sanity", check_date_sanity)
    reg.register("party_identifiers", check_party_identifiers)
    return reg


def validate_document(
    doc: ExtractedDocument,
    *,
    settings: Settings | None = None,
    registry: RuleRegistry | None = None,
) -> ValidationReport:
    """Run all registered validation rules against `doc` and return a `ValidationReport`."""
    active_registry = registry or build_default_registry()
    return active_registry.run(doc, settings=settings)
