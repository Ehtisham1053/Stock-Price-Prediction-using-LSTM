# app/services/data_fetch.py
from __future__ import annotations

import time
from typing import Iterable, Optional, Sequence, Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from app.core.config import Config
from app.utils.dates import resolve_dates, align_index_to_calendar

# ---------- internal helpers ----------

def _flatten_columns(cols, ticker_expected: Optional[str] = None) -> Sequence[str]:
    if isinstance(cols, pd.MultiIndex):
        if cols.nlevels == 2:
            secs = {t for (_, t) in cols}
            if len(secs) == 1 and (ticker_expected is None or list(secs)[0] == ticker_expected):
                flat = [str(f) for (f, _) in cols]
            else:
                flat = ["_".join([str(x) for x in tup if x is not None]) for tup in cols]
        else:
            flat = ["_".join([str(x) for x in tup if x is not None]) for tup in cols]
    else:
        flat = [str(c) for c in cols]
    return [c.strip().title().replace(" ", "_") for c in flat]

def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _postprocess_df(
    df: pd.DataFrame,
    *,
    ticker_up: str,
    calendar: str,
    required_cols: Optional[Iterable[str]],
    start_s: str,
    end_s: str,
) -> pd.DataFrame:
    # index hygiene
    df.index = pd.to_datetime(df.index, utc=False)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df = df.loc[~df.index.duplicated(keep="first")].sort_index()

    # flatten + numeric
    df.columns = _flatten_columns(df.columns, ticker_expected=ticker_up)
    df = _coerce_numeric(df).dropna(how="all")

    # align to exchange calendar
    aligned_idx = align_index_to_calendar(df.index, start_s, end_s, calendar)
    df = df.loc[aligned_idx]

    # ensure required columns
    if required_cols:
        req = set([c.strip().title().replace(" ", "_") for c in required_cols])
        have = set(df.columns)
        missing = req - have
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    keep = [c for c in ["Open", "High", "Low", "Close", "Adj_Close", "Volume"] if c in df.columns]
    df = df[keep].dropna(subset=[c for c in ["Open", "High", "Low", "Close"] if c in keep])
    if df.empty:
        raise ValueError("All rows dropped after cleaning.")
    return df

# ---------- public API ----------

def fetch_ohlcv_yf(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    *,
    calendar: str = Config.CALENDAR,
    auto_adjust: bool = True,
    max_retries: int = 3,
    sleep_sec: float = 1.5,
    required_cols: Optional[Iterable[str]] = None,
    fallback_period: str = "max",
    alt_years: int = 10,
) -> pd.DataFrame:
    """
    Robust yfinance fetch:
      1) Try explicit [start,end) window with retries + backoff
      2) Fallback A: shrink window to last `alt_years`
      3) Fallback B: use period-based download (e.g., period='max')
    Always calendar-aligns and standardizes columns.
    """
    start_dt, end_dt = resolve_dates(start, end)
    start_s, end_s = str(start_dt.date()), str(end_dt.date())
    ticker_up = ticker.upper().strip()

    last_err: Optional[Exception] = None

    # --- Attempt 1: windowed fetch with retries ---
    for attempt in range(1, max_retries + 1):
        try:
            df = yf.download(
                tickers=ticker_up,
                start=start_s,
                end=end_s,
                auto_adjust=auto_adjust,
                progress=False,
                threads=True,
                group_by="column",
            )
            if df is None or df.empty:
                raise ValueError(f"No data for {ticker_up} in {start_s}→{end_s}")
            return _postprocess_df(df, ticker_up=ticker_up, calendar=calendar,
                                   required_cols=required_cols, start_s=start_s, end_s=end_s)
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(sleep_sec * attempt)  # simple backoff
            else:
                break  # go to fallbacks

    # --- Fallback A: shrink window to last alt_years ---
    try:
        end_dt2 = end_dt
        start_dt2 = end_dt2 - pd.DateOffset(years=max(2, int(alt_years)))
        start_s2, end_s2 = str(start_dt2.date()), str(end_dt2.date())

        df2 = yf.download(
            tickers=ticker_up,
            start=start_s2,
            end=end_s2,
            auto_adjust=auto_adjust,
            progress=False,
            threads=True,
            group_by="column",
        )
        if df2 is not None and not df2.empty:
            return _postprocess_df(df2, ticker_up=ticker_up, calendar=calendar,
                                   required_cols=required_cols, start_s=start_s2, end_s=end_s2)
    except Exception as e2:
        last_err = e2

    # --- Fallback B: period-based ---
    try:
        df3 = yf.download(
            tickers=ticker_up,
            period=fallback_period,   # e.g. '1y','5y','max'
            interval="1d",
            auto_adjust=auto_adjust,
            progress=False,
            threads=True,
            group_by="column",
        )
        if df3 is None or df3.empty:
            raise ValueError(f"No data for {ticker_up} with period={fallback_period}")

        # derive window from data for calendar alignment
        s0 = str(pd.to_datetime(df3.index).min().date())
        e0 = str((pd.to_datetime(df3.index).max() + pd.Timedelta(days=1)).date())
        return _postprocess_df(df3, ticker_up=ticker_up, calendar=calendar,
                               required_cols=required_cols, start_s=s0, end_s=e0)
    except Exception as e3:
        last_err = e3
        raise RuntimeError(
            f"Data fetch failed after {max_retries} attempts and fallbacks: {last_err}"
        ) from last_err


def moving_averages(df: pd.DataFrame, col: str, windows: Iterable[int]) -> Dict[int, pd.Series]:
    assert col in df.columns, f"{col} not in DataFrame"
    out = {}
    for w in windows:
        out[w] = df[col].rolling(window=w, min_periods=max(1, w // 3)).mean()
    return out
