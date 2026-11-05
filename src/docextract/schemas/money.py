"""Exact decimal monetary representation and currency-safe arithmetic."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator


class Currency(StrEnum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    PHP = "PHP"
    JPY = "JPY"
    CAD = "CAD"
    AUD = "AUD"
    SGD = "SGD"


ZERO_DECIMAL_CURRENCIES = frozenset({Currency.JPY})


def quantize_amount(
    value: Decimal | str | int | float, currency: Currency = Currency.USD
) -> Decimal:
    """Quantize a monetary amount using standard half-up financial rounding."""
    try:
        dec = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid monetary amount: {value!r}") from exc

    exponent = Decimal("1") if currency in ZERO_DECIMAL_CURRENCIES else Decimal("0.01")
    return dec.quantize(exponent, rounding=ROUND_HALF_UP)


class Money(BaseModel):
    """Immutable monetary value backed by `Decimal` and an ISO 4217 currency code."""

    model_config = ConfigDict(frozen=True)

    amount: Decimal
    currency: Currency = Currency.USD

    @field_validator("currency", mode="before")
    @classmethod
    def _normalize_currency(cls, v: Any) -> Currency:
        if isinstance(v, Currency):
            return v
        if isinstance(v, str):
            return Currency(v.strip().upper())
        raise ValueError(f"Unsupported currency: {v!r}")

    @field_validator("amount", mode="before")
    @classmethod
    def _parse_decimal(cls, v: Any) -> Decimal:
        if isinstance(v, float):
            # Convert via str to avoid binary floating point representation artifacts
            v = str(v)
        try:
            return Decimal(v)
        except (InvalidOperation, ValueError, TypeError) as exc:
            raise ValueError(f"Cannot convert {v!r} to Decimal") from exc

    def quantized(self) -> Money:
        return Money(amount=quantize_amount(self.amount, self.currency), currency=self.currency)

    def _assert_same_currency(self, other: Money) -> None:
        if self.currency != other.currency:
            raise ValueError(
                f"Currency mismatch: cannot combine {self.currency.value} "
                f"and {other.currency.value}"
            )

    def __add__(self, other: Money) -> Money:
        self._assert_same_currency(other)
        return Money(
            amount=quantize_amount(self.amount + other.amount, self.currency),
            currency=self.currency,
        )

    def __sub__(self, other: Money) -> Money:
        self._assert_same_currency(other)
        return Money(
            amount=quantize_amount(self.amount - other.amount, self.currency),
            currency=self.currency,
        )

    def multiply(self, factor: Decimal | int | str) -> Money:
        dec_factor = Decimal(str(factor))
        return Money(
            amount=quantize_amount(self.amount * dec_factor, self.currency),
            currency=self.currency,
        )

    def within_tolerance(self, other: Money, tolerance: Decimal = Decimal("0.02")) -> bool:
        self._assert_same_currency(other)
        return abs(self.amount - other.amount) <= tolerance

    def format(self) -> str:
        q = quantize_amount(self.amount, self.currency)
        return f"{self.currency.value} {q}"
