# app/services/validators.py
from __future__ import annotations

import re
from typing import Any, Dict, Tuple

from app.core.config import Config

# Accept Yahoo-style tickers: letters/digits plus . - ^ =
_TICKER_RE = re.compile(r"^[A-Za-z0-9.\-^=]{1,15}$")
_HEADS = {"close", "open", "high"}

class ValidationError(ValueError):
    """Raised when request input fails validation."""

# --- helpers ---

def _normalize_head(head: Any) -> str:
    if not isinstance(head, str):
        raise ValidationError("`head` must be a string: close | open | high")
    h = head.strip().lower()
    if h not in _HEADS:
        raise ValidationError("`head` must be one of: close | open | high")
    return h

# gentle auto-fix for very common typos (you asked for minimal friction)
_COMMON_TICKER_FIXES = {
    "APPL": "AAPL",
}

def _normalize_ticker(ticker: Any) -> str:
    if not isinstance(ticker, str):
        raise ValidationError("`ticker` must be a string.")
    t = ticker.strip().upper()
    t = _COMMON_TICKER_FIXES.get(t, t)
    if not _TICKER_RE.match(t):
        raise ValidationError("`ticker` contains invalid characters or length.")
    return t

def _normalize_days(days: Any, *, max_days: int = Config.MAX_FUTURE_DAYS) -> int:
    try:
        if isinstance(days, str):
            days = days.strip()
            if days == "":
                raise ValueError
        n = int(days)
    except Exception:
        raise ValidationError("`days` must be an integer.")
    if not (1 <= n <= max_days):
        raise ValidationError(f"`days` must be between 1 and {max_days}.")
    return n

# --- public API ---

def validate_predict_payload(payload: Dict[str, Any]) -> Tuple[str, str, int]:
    """
    Expecting:
      {
        "head": "close" | "open" | "high",
        "ticker": "AAPL" | "MSFT" | ...,
        "days": 1..MAX_FUTURE_DAYS
      }
    Returns normalized (head, ticker, days) or raises ValidationError.
    """
    if not isinstance(payload, dict):
        raise ValidationError("Invalid JSON payload.")
    head   = _normalize_head(payload.get("head"))
    ticker = _normalize_ticker(payload.get("ticker"))
    days   = _normalize_days(payload.get("days"))
    return head, ticker, days

def validate_analysis_query(ticker_raw: Any) -> str:
    """
    For /api/analysis?ticker=...
    """
    return _normalize_ticker(ticker_raw)
