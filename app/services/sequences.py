# app/services/sequences.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Literal, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from app.core.config import Config
from app.services.data_fetch import fetch_ohlcv_yf
from app.utils.dates import resolve_dates

Head = Literal["close", "open", "high"]
logger = logging.getLogger(__name__)

# ---------------------------
# Data containers
# ---------------------------

@dataclass
class SplitFrames:
    train: pd.DataFrame
    val:   pd.DataFrame
    test:  pd.DataFrame

@dataclass
class ScaledFrames:
    train: pd.DataFrame
    val:   pd.DataFrame
    test:  pd.DataFrame
    scaler: StandardScaler

@dataclass
class SequenceArrays:
    X_train: np.ndarray
    y_train: np.ndarray
    pc_train: np.ndarray  # Prev_Close aligned with y_train timestamps

    X_val:   np.ndarray
    y_val:   np.ndarray
    pc_val:  np.ndarray

    X_test:  np.ndarray
    y_test:  np.ndarray
    pc_test: np.ndarray

@dataclass
class PrepResult:
    # trimmed raw price frame used for this head
    raw_prices: pd.DataFrame              # subset of OHLC used (e.g., Close/Open/High)
    target_df:  pd.DataFrame              # columns: Prev_Close, <TargetCol> (+ optional actual price)

    # splits / scaled / sequences
    splits:     SplitFrames
    scaled:     ScaledFrames
    seq:        SequenceArrays

    # metadata
    ticker:     str
    head:       Head
    target_col: str                       # "LogRet" | "OvernightLogRet" | "HighLogRet"
    lookback:   int
    trim_years: int
    calendar:   str


# ---------------------------
# Target builders (mirrors notebooks)
# ---------------------------

def _build_close_target(ohlc: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Close returns:
      LogRet_t = ln(Close_t / Close_{t-1})
      reconstruct: Ĉ_t = Prev_Close_t * exp(LogRet_t)
    """
    close = ohlc["Close"].copy()
    prev_close = close.shift(1)
    logret = np.log(close / prev_close)
    df = pd.DataFrame({
        "Close": close,
        "Prev_Close": prev_close,
        "LogRet": logret
    }).dropna()
    return df, "LogRet"

def _build_open_target(ohlc: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Overnight (Open) returns:
      OvernightLogRet_t = ln(Open_t / Close_{t-1})
      reconstruct: Ô_t = Prev_Close_t * exp(OvernightLogRet_t)
    """
    open_ = ohlc["Open"].copy()
    prev_close = ohlc["Close"].shift(1)
    logret = np.log(open_ / prev_close)
    df = pd.DataFrame({
        "Open": open_,
        "Prev_Close": prev_close,
        "OvernightLogRet": logret
    }).dropna()
    return df, "OvernightLogRet"

def _build_high_target(ohlc: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    High returns:
      HighLogRet_t = ln(High_t / Close_{t-1})
      reconstruct: Ĥ_t = Prev_Close_t * exp(HighLogRet_t)
    """
    high = ohlc["High"].copy()
    prev_close = ohlc["Close"].shift(1)
    logret = np.log(high / prev_close)
    df = pd.DataFrame({
        "High": high,
        "Prev_Close": prev_close,
        "HighLogRet": logret
    }).dropna()
    return df, "HighLogRet"


# ---------------------------
# Split / scale / sequences
# ---------------------------

def _split_chrono(df: pd.DataFrame, train_ratio=0.70, val_ratio=0.15) -> SplitFrames:
    n = len(df)
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))
    return SplitFrames(
        train=df.iloc[:train_end].copy(),
        val=df.iloc[train_end:val_end].copy(),
        test=df.iloc[val_end:].copy(),
    )

def _scale_train_only(
    splits: SplitFrames,
    target_col: str,
    existing_scaler: Optional[StandardScaler] = None
) -> ScaledFrames:
    """
    If existing_scaler is provided (AAPL base), reuse it for all splits.
    Otherwise, fit a new StandardScaler on TRAIN only (non-AAPL finetune path).
    """
    if existing_scaler is None:
        scaler = StandardScaler()
        scaler.fit(splits.train[[target_col]])
    else:
        scaler = existing_scaler

    def _apply(d: pd.DataFrame) -> pd.DataFrame:
        arr = scaler.transform(d[[target_col]])
        return pd.DataFrame(arr, index=d.index, columns=[target_col])

    return ScaledFrames(
        train=_apply(splits.train),
        val=_apply(splits.val),
        test=_apply(splits.test),
        scaler=scaler
    )

def _make_sequences_std(
    scaled: ScaledFrames,
    splits: SplitFrames,
    target_col: str,
    lookback: int
) -> SequenceArrays:
    """
    Create rolling windows of standardized returns with Prev_Close aligned to the target time t.
    Shapes are enforced to (N, lookback, 1).
    """
    def _seq(data_sc: pd.DataFrame, prev_close_series: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X, y, pc = [], [], []
        vals = data_sc[target_col].values.astype(np.float32)
        prev_vals = prev_close_series.reindex(data_sc.index).values.astype(np.float32)
        for i in range(lookback, len(vals)):
            X.append(vals[i - lookback:i].reshape(lookback, 1))
            y.append(vals[i])
            pc.append(prev_vals[i])
        X = np.asarray(X, np.float32)
        y = np.asarray(y, np.float32)
        pc = np.asarray(pc, np.float32)
        if X.ndim != 3 or X.shape[1] != lookback or X.shape[2] != 1:
            raise ValueError(
                f"Sequence build error: expected X shape (N,{lookback},1) but got {X.shape}. "
                f"Check lookback and reshape logic."
            )
        return X, y, pc

    Xtr, ytr, pctr = _seq(scaled.train, splits.train["Prev_Close"])
    Xva, yva, pcva = _seq(scaled.val,   splits.val["Prev_Close"])
    Xte, yte, pcte = _seq(scaled.test,  splits.test["Prev_Close"])

    # Guards
    if not np.all(np.isfinite(Xtr)) or not np.all(np.isfinite(ytr)):
        raise ValueError("Non-finite values detected in training sequences.")
    if not np.all(pctr > 0):
        raise ValueError("Prev_Close must be positive to reconstruct prices.")

    return SequenceArrays(
        X_train=Xtr, y_train=ytr, pc_train=pctr,
        X_val=Xva,   y_val=yva,   pc_val=pcva,
        X_test=Xte,  y_test=yte,  pc_test=pcte,
    )


# ---------------------------
# Public prep API
# ---------------------------

def prepare_sequences_for_head(
    ticker: str,
    head: Head,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    use_base_scaler: Optional[StandardScaler] = None,
    lookback: Optional[int] = None,
    trim_years: Optional[int] = None,
    calendar: Optional[str] = None,
) -> PrepResult:
    """
    Fetches OHLCV, trims to last TRIM_YEARS, builds the chosen target (close|open|high),
    splits 70/15/15, applies StandardScaler (TRAIN-only unless use_base_scaler is provided),
    and returns lookback sequences with Prev_Close aligned to t (for reconstruction).

    IMPORTANT:
      - The caller (ModelManager) must pass lookback inferred from the loaded model:
          expected_lb = int(model.input_shape[1])
          prepare_sequences_for_head(..., lookback=expected_lb)
      - This ensures (N, lookback, 1) matches the model’s input.
    """
    ticker = ticker.upper().strip()
    if lookback is None:
        # Hard fail: we shouldn't silently use Config.LOOKBACK, to avoid shape mismatches.
        raise ValueError("prepare_sequences_for_head: 'lookback' must be provided (use model.input_shape[1]).")
    lb = int(lookback)
    ty = int(trim_years or Config.TRIM_YEARS)
    cal = calendar or Config.CALENDAR

    start_dt, end_dt = resolve_dates(start, end)
    ohlc_full = fetch_ohlcv_yf(
        ticker=ticker, start=str(start_dt.date()), end=str(end_dt.date()),
        calendar=cal, auto_adjust=True, required_cols=["Open", "High", "Low", "Close"]
    )

    # Trim to last N years (mirror notebooks)
    cutoff = ohlc_full.index.max() - pd.DateOffset(years=ty)
    ohlc_trim = ohlc_full[ohlc_full.index >= cutoff].copy()

    # Build target per head
    if head == "close":
        target_df, target_col = _build_close_target(ohlc_trim)
        raw_prices = ohlc_trim[["Close"]]
    elif head == "open":
        target_df, target_col = _build_open_target(ohlc_trim)
        raw_prices = ohlc_trim[["Open", "Close"]]
    elif head == "high":
        target_df, target_col = _build_high_target(ohlc_trim)
        raw_prices = ohlc_trim[["High", "Close"]]
    else:
        raise ValueError("head must be one of: 'close', 'open', 'high'")

    # Guards similar to notebooks
    min_needed = lb + 200
    if len(target_df) < min_needed:
        raise ValueError(
            f"Not enough rows after trim/diff for {ticker}/{head}: have {len(target_df)}, need ≥ {min_needed}."
        )

    assert isinstance(target_df.index, pd.DatetimeIndex) and target_df.index.tz is None
    assert target_df.index.is_monotonic_increasing
    if not np.isfinite(target_df[target_col]).all():
        raise ValueError("Non-finite values in target returns.")

    # Splits
    splits = _split_chrono(target_df)

    # Scale (TRAIN-only) or use provided base scaler
    scaled = _scale_train_only(splits, target_col, existing_scaler=use_base_scaler)

    # Sequences
    seq = _make_sequences_std(scaled, splits, target_col, lb)

    # Debug logs to help diagnose future mismatches quickly
    logger.info(
        "Prepared %s/%s: lookback=%d | X_train=%s X_val=%s X_test=%s",
        ticker, head, lb, seq.X_train.shape, seq.X_val.shape, seq.X_test.shape
    )

    return PrepResult(
        raw_prices=raw_prices,
        target_df=target_df,
        splits=splits,
        scaled=scaled,
        seq=seq,
        ticker=ticker,
        head=head,
        target_col=target_col,
        lookback=lb,
        trim_years=ty,
        calendar=cal,
    )


# ---------------------------
# Convenience: compute MAs for analysis route
# ---------------------------

def moving_average_frames(ohlc: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Build a small bundle of MA overlays used in the analysis page.
    Returns dict with:
      - 'close_ma': DataFrame with Close, MA30, MA50
      - 'open_ma' : DataFrame with Open,  MA30, MA50
      - 'high_ma' : DataFrame with High,  MA30, MA50
    """
    out: Dict[str, pd.DataFrame] = {}
    for col, key in [("Close", "close_ma"), ("Open", "open_ma"), ("High", "high_ma")]:
        if col not in ohlc.columns:
            continue
        base = pd.DataFrame({col: ohlc[col]})
        base["MA30"] = ohlc[col].rolling(window=30, min_periods=10).mean()
        base["MA50"] = ohlc[col].rolling(window=50, min_periods=15).mean()
        out[key] = base
    return out
