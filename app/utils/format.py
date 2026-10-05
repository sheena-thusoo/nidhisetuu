"""Deterministic display formatting (Indian digit grouping, money strings)."""

from __future__ import annotations

import time
import uuid
from decimal import ROUND_HALF_UP, Decimal

MONEY_QUANT = Decimal("0.01")


def quantize_money(value: Decimal | int | float | str) -> Decimal:
    return Decimal(str(value)).quantize(MONEY_QUANT, rounding=ROUND_HALF_UP)


def _group_indian(int_part: str) -> str:
    """1234567 -> 12,34,567"""
    if len(int_part) <= 3:
        return int_part
    head, tail = int_part[:-3], int_part[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts + [tail])


def format_inr(value: Decimal | int | float | str | None, with_symbol: bool = True) -> str | None:
    if value is None:
        return None
    dec = Decimal(str(value))
    negative = dec < 0
    dec = -dec if negative else dec
    rounded = dec.quantize(MONEY_QUANT, rounding=ROUND_HALF_UP)
    int_part, _, frac_part = f"{rounded:f}".partition(".")
    frac = (frac_part + "00")[:2]
    frac_str = "" if frac == "00" else f".{frac}"
    sign = "-" if negative else ""
    symbol = "\u20b9" if with_symbol else ""
    return f"{sign}{symbol}{_group_indian(int_part)}{frac_str}"


def format_number(value: Decimal | int | float | str | None) -> str | None:
    return format_inr(value, with_symbol=False)


def new_id() -> str:
    return uuid.uuid4().hex


def short_id() -> str:
    return uuid.uuid4().hex[:12]
