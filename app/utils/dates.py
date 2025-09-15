# app/utils/dates.py
import pandas as pd
import pandas_market_calendars as mcal
from typing import Optional, Tuple

def last_completed_utc_day() -> pd.Timestamp:
    """Yesterday at 00:00 UTC (naive)."""
    return (pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=1))

def utc_midnight_naive(ts) -> pd.Timestamp:
    """Normalize any ts to midnight UTC and return tz-naive timestamp."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None or ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.normalize().tz_localize(None)

def resolve_dates(start: Optional[str] = None, end: Optional[str] = None) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Return (start, end) at UTC midnight (tz-naive)."""
    s = pd.Timestamp(start) if start else pd.Timestamp("2000-01-01")
    e = pd.Timestamp(end)   if end   else last_completed_utc_day()
    return utc_midnight_naive(s), utc_midnight_naive(e)

def trading_index(start_date: str, end_date: str, calendar: str) -> pd.DatetimeIndex:
    """XNYS etc. -> tz-naive trading days between [start_date, end_date]."""
    cal = mcal.get_calendar(calendar)
    sched = cal.schedule(start_date=start_date, end_date=end_date)
    idx = sched.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    return pd.DatetimeIndex(idx)

def align_index_to_calendar(df_idx: pd.DatetimeIndex, start_date: str, end_date: str, calendar: str) -> pd.DatetimeIndex:
    """Intersect a DataFrame index with the exchange trading index."""
    cal_idx = trading_index(start_date, end_date, calendar)
    return df_idx.intersection(cal_idx)

def future_trading_days(last_date: pd.Timestamp, n: int, calendar: str) -> pd.DatetimeIndex:
    """
    Next n trading days after last_date, using exchange calendar.
    Uses a loose window (≈ 3n days) to ensure enough sessions.
    """
    cal = mcal.get_calendar(calendar)
    start_d = (pd.Timestamp(last_date) + pd.Timedelta(days=1)).date()
    end_d   = (pd.Timestamp(last_date) + pd.DateOffset(days=max(10, n * 3))).date()
    sched = cal.schedule(start_date=str(start_d), end_date=str(end_d))
    idx = sched.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    idx = pd.DatetimeIndex(idx)
    if len(idx) < n:
        raise ValueError("Not enough future trading days from calendar.")
    return idx[:n]
