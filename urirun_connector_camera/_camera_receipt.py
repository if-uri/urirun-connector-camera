"""Receipt (paragon) text parsing helpers extracted from core.py."""
from __future__ import annotations

import re
from typing import Any

_PRICE_RE = re.compile(r"(\d{1,3}(?:[  ]\d{3})*|\d+)[.,](\d{2})(?!\d)")
_TOTAL_KEYS = ("suma", "razem", "do zaplaty", "do zaplaty", "total", "lacznie", "summa", "naleznosc")
_DATE_RE = re.compile(r"(\d{4}[-./]\d{2}[-./]\d{2}|\d{2}[-./]\d{2}[-./]\d{4})")
_NIP_RE = re.compile(r"NIP[:\s]*([0-9][0-9\- \t]{8,14})", re.IGNORECASE)


def _fold(text: str) -> str:
    """Lowercase and strip Polish diacritics so keyword matching is robust to OCR/locale."""
    table = str.maketrans("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ", "acelnoszzACELNOSZZ")
    return text.translate(table).lower()


def _to_amount(whole: str, cents: str) -> float:
    return round(int(whole.replace(" ", "").replace(" ", "")) + int(cents) / 100.0, 2)


def _parse_receipt(text: str) -> dict[str, Any]:
    """Turn raw receipt OCR text into structured data: line items (name + price), the total,
    currency, date and NIP. Heuristic and locale-tolerant (PL/EN) — every field is best
    effort and may be null when the print/OCR is too noisy."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    items: list[dict[str, Any]] = []
    total: float | None = None
    for line in lines:
        prices = _PRICE_RE.findall(line)
        if not prices:
            continue
        amount = _to_amount(*prices[-1])           # the rightmost money token on the line
        folded = _fold(line)
        if any(key in folded for key in _TOTAL_KEYS):
            total = amount                          # a total/sum line, not a product
            continue
        name = _PRICE_RE.sub("", line).strip(" .:-xX*\t")
        name = re.sub(r"\s{2,}", " ", name)
        if len(name) >= 2:
            items.append({"name": name, "price": amount})

    currency = None
    for cur, pat in (("PLN", r"\bpln\b|z[lł]\b"), ("EUR", r"\beur\b|€"), ("USD", r"\busd\b|\$")):
        if re.search(pat, _fold(text)):
            currency = cur
            break
    date_m = _DATE_RE.search(text or "")
    nip_m = _NIP_RE.search(text or "")
    items_sum = round(sum(i["price"] for i in items), 2)
    total_source = "total-line" if total is not None else None
    if total is None and items:                     # fall back to the largest amount seen
        total = max(i["price"] for i in items)
        total_source = "max-item"
    return {
        "items": items,
        "itemCount": len(items),
        "total": total,
        "totalSource": total_source,
        "itemsSum": items_sum,
        "currency": currency,
        "date": date_m.group(1) if date_m else None,
        "nip": re.sub(r"\D", "", nip_m.group(1))[:10] if nip_m else None,
        "lines": len(lines),
    }
