# app/services/forecaster.py
from __future__ import annotations

import logging
from typing import Optional, Dict
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from app.core.config import Config
from app.services.model_manager import ModelManager, ManagerRun
from app.utils.dates import future_trading_days

logger = logging.getLogger(__name__)


# ---------------------------
# Helpers
# ---------------------------

def _infer_lb_from_model(model) -> int:
    ishape = getattr(model, "input_shape", None)
    if not ishape or len(ishape) < 3:
        raise ValueError(f"Unexpected model.input_shape: {ishape}")
    lb = int(ishape[1])
    ch = int(ishape[2])
    if ch != 1:
        raise ValueError(f"Model expects {ch} channels; pipeline assumes 1.")
    if lb <= 0:
        raise ValueError(f"Invalid lookback inferred from model.input_shape={ishape}")
    return lb


# ---------------------------
# Internal: prev-Close path via Close base
# ---------------------------

def _try_prev_close_path_via_close_base(
    mm: ModelManager,
    close_series: pd.Series,  # trimmed close prices for the *requested ticker*
    n_steps: int,
) -> Optional[np.ndarray]:
    """
    Use the AAPL Close base model + scaler to simulate a prev-close path for the ticker.
    - Standardize the ticker's Close log-returns with the base Close scaler.
    - Seed last LOOKBACK standardized values (inferred from base model), predict recursively.
    - Convert each standardized step back to log-return, evolve close, and return prev-close path:
        prev_path[0] = last_close
        prev_path[k] = predicted Close at step k-1  (for k>=1)
    """
    try:
        base = mm._get_base_bundle("close")  # accessing internal cache (intentional)
    except Exception as e:
        logger.warning("Close base bundle not available: %s", e)
        return None

    try:
        lookback = _infer_lb_from_model(base.model)
    except Exception as e:
        logger.warning("Could not infer lookback from close base model: %s", e)
        return None

    # Build & standardize close log-returns using base close scaler
    lr = np.log(close_series).diff().dropna()
    if len(lr) < lookback + 5:
        logger.warning("Not enough close history to seed prev-close path (need >= lookback+5).")
        return None

    # Standardize with base scaler
    std_vals = base.scaler.transform(lr.values.reshape(-1, 1)).ravel().astype(np.float32)
    # Seed window shape -> (1, lookback, 1)
    cur = std_vals[-lookback:].reshape(1, lookback, 1)
    last_close = float(close_series.iloc[-1])

    preds_close = []
    prev = last_close
    for _ in range(n_steps):
        y_std = float(base.model.predict(cur, verbose=0).ravel()[0])
        # back to log-return with the same scaler used to standardize
        y_log = float(base.scaler.inverse_transform(np.array([[y_std]], dtype=np.float32)).ravel()[0])
        nxt = prev * np.exp(y_log)
        preds_close.append(nxt)

        # roll window left by 1 and place y_std at the end
        cur[:, :-1, :] = cur[:, 1:, :]
        cur[0, -1, 0] = y_std
        prev = nxt

    prev_path = np.empty(n_steps, dtype=np.float32)
    prev_path[0] = last_close
    if n_steps > 1:
        prev_path[1:] = np.asarray(preds_close[:-1], dtype=np.float32)
    return prev_path


def _prev_close_path_fallback(close_series: pd.Series, n_steps: int) -> np.ndarray:
    """Median-drift proxy when Close base is not available."""
    lr = np.log(close_series).diff().dropna()
    drift = float(np.median(lr.tail(120))) if len(lr) else 0.0
    prev = float(close_series.iloc[-1])
    out = np.empty(n_steps, dtype=np.float32)
    for i in range(n_steps):
        if i == 0:
            out[i] = prev
        else:
            prev = prev * np.exp(drift)
            out[i] = prev
    return out


# ---------------------------
# Validation on price scale (Val split)
# ---------------------------

def validate_on_price(run: ManagerRun) -> Dict[str, float]:
    """
    Compute price-scale metrics on the validation split by reconstructing prices:
      Close: Ĉ_t = Prev_Close_t * exp( r̂_close_t )
      Open : Ô_t = Prev_Close_t * exp( r̂_overnight_t )
      High : Ĥ_t = Prev_Close_t * exp( r̂_high_t )
    Returns dict with MAE, RMSE, MAPE, R2, Directional Accuracy (% vs Prev_Close).
    """
    model = run.model
    scaler: StandardScaler = run.scaler
    seq = run.prep.seq

    # 1) Predict standardized returns
    y_pred_std = model.predict(seq.X_val, verbose=0).ravel().astype(np.float32)
    y_true_std = seq.y_val.ravel().astype(np.float32)

    # 2) Back to log-returns
    y_pred_log = scaler.inverse_transform(y_pred_std.reshape(-1, 1)).ravel()
    y_true_log = scaler.inverse_transform(y_true_std.reshape(-1, 1)).ravel()

    # 3) Reconstruct price from Prev_Close_t
    pred_price = seq.pc_val * np.exp(y_pred_log)
    true_price = seq.pc_val * np.exp(y_true_log)

    # 4) Metrics
    mae  = float(mean_absolute_error(true_price, pred_price))
    rmse = float(mean_squared_error(true_price, pred_price))
    mape = float((np.abs((true_price - pred_price) / np.maximum(1e-8, np.abs(true_price)))).mean() * 100.0)
    r2   = float(r2_score(true_price, pred_price))
    da   = float((np.sign(pred_price - seq.pc_val) == np.sign(true_price - seq.pc_val)).mean() * 100.0)

    return {
        "mae": mae,
        "rmse": rmse,
        "mape_pct": mape,
        "r2": r2,
        "directional_acc_pct": da,
    }


# ---------------------------
# N-day forecast (recursive rollout)
# ---------------------------

def _combined_std_series(run: ManagerRun) -> pd.Series:
    """Concat train/val/test standardized series for seeding the last LOOKBACK points."""
    tcol = run.prep.target_col
    sc   = run.prep.scaled
    s = pd.concat([sc.train[tcol], sc.val[tcol], sc.test[tcol]]).sort_index()
    return s


def forecast_n_days(
    *,
    mm: ModelManager,
    run: ManagerRun,
    n_future: int
) -> pd.DataFrame:
    """
    Produce a DataFrame indexed by future trading days with predicted prices:
      - head == 'close' → column: 'pred_close'
      - head == 'open'  → column: 'pred_open'
      - head == 'high'  → column: 'pred_high'
    Includes helper columns: 'prev_close_used' and 'pred_logret'.
    """
    assert 1 <= n_future <= Config.MAX_FUTURE_DAYS, "n_future out of bounds"

    model = run.model
    scaler: StandardScaler = run.scaler
    head = run.head
    lookback = run.prep.lookback

    std_series = _combined_std_series(run)
    if len(std_series) < lookback:
        raise ValueError("Not enough history to seed the forecast window.")

    # Seed standardized window as (1, lookback, 1)
    cur = std_series.values[-lookback:].astype(np.float32).reshape(1, lookback, 1)

    # Future trading days from last available date in target_df
    last_date = run.prep.target_df.index.max()
    idx = future_trading_days(last_date, n_future, calendar=run.prep.calendar)

    if head == "close":
        # Evolve the close itself
        last_close = float(run.prep.raw_prices["Close"].iloc[-1])
        prev = last_close
        prev_used, pred_log, pred_price = [], [], []

        for _ in range(n_future):
            y_std = float(model.predict(cur, verbose=0).ravel()[0])
            y_log = float(scaler.inverse_transform(np.array([[y_std]], dtype=np.float32)).ravel()[0])
            nxt   = prev * np.exp(y_log)

            prev_used.append(prev); pred_log.append(y_log); pred_price.append(nxt)
            # roll window & set last
            cur[:, :-1, :] = cur[:, 1:, :]
            cur[0, -1, 0] = y_std


            
            prev = nxt

        out = pd.DataFrame({
            "prev_close_used": prev_used,
            "pred_logret": pred_log,
            "pred_close": pred_price,
        }, index=idx)

    else:
        # For open/high we need a prev-close path
        # Prefer target_df['Prev_Close']; fall back to any 'Close' in raw_prices
        if "Prev_Close" in run.prep.target_df.columns:
            close_series = run.prep.target_df["Prev_Close"]
        else:
            close_series = run.prep.raw_prices.get("Close", pd.Series(dtype=float))
        close_series = close_series.dropna()
        if close_series.empty:
            raise ValueError("No prev-close series available to reconstruct prices.")

        prev_path = _try_prev_close_path_via_close_base(mm, close_series, n_future)
        if prev_path is None:
            prev_path = _prev_close_path_fallback(close_series, n_future)

        pred_log, pred_price = [], []
        for k in range(n_future):
            prev_close_k = float(prev_path[k])
            y_std = float(model.predict(cur, verbose=0).ravel()[0])
            y_log = float(scaler.inverse_transform(np.array([[y_std]], dtype=np.float32)).ravel()[0])
            price = prev_close_k * np.exp(y_log)

            pred_log.append(y_log); pred_price.append(price)
            cur[:, :-1, :] = cur[:, 1:, :]
            cur[0, -1, 0] = y_std

        col_name = "pred_open" if head == "open" else "pred_high"
        out = pd.DataFrame({
            "prev_close_used": prev_path,
            "pred_logret": pred_log,
            col_name: pred_price,
        }, index=idx)

    out.index.name = "date"
    return out
