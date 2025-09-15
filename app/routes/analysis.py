# app/routes/analysis.py
from __future__ import annotations
import pandas as pd
from flask import Blueprint, request, jsonify
import logging
import pandas as pd
from app.core.config import Config
from app.services.validators import validate_analysis_query, ValidationError
from app.services.data_fetch import fetch_ohlcv_yf
from app.services.sequences import moving_average_frames

bp = Blueprint("analysis", __name__)
log = logging.getLogger(__name__)

@bp.get("/analysis")
def analysis():
    """GET /api/analysis?ticker=TSLA
    Returns:
      - first_5_rows (tabular)
      - chart series for:
          MA30/50 vs Close
          MA30/50 vs Open
          MA30/50 vs High
          raw Close
          raw Open
    """
    try:
        ticker = validate_analysis_query(request.args.get("ticker", "AAPL"))
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400

    try:
        # Limit window to reduce payload & avoid yfinance timeouts on huge ranges
        years = getattr(Config, "ANALYSIS_YEARS", 10)
        end_dt = pd.Timestamp.utcnow().normalize()
        start_dt = end_dt - pd.DateOffset(years=int(years))
        start_s = str(start_dt.date())

        # First attempt: explicit [start_s, end) window (robust fetch has its own retries/backoffs)
        ohlc = fetch_ohlcv_yf(
            ticker=ticker,
            start=start_s,
            end=None,  # last completed UTC day
            calendar=Config.CALENDAR,
            auto_adjust=True,
            required_cols=["Open", "High", "Low", "Close"],
            alt_years=int(years),
        )

    except Exception as e1:
        # Fallback: period-based fetch (smaller fixed period to avoid giant downloads)
        try:
            ohlc = fetch_ohlcv_yf(
                ticker=ticker,
                start=None,
                end=None,
                calendar=Config.CALENDAR,
                auto_adjust=True,
                required_cols=["Open", "High", "Low", "Close"],
                fallback_period="5y",   # try 5 years if windowed fetch failed
                alt_years=int(years),
            )
        except Exception as e2:
            log.exception("Analysis fetch failed")
            return jsonify({"error": f"Analysis failed: {e2}"}), 500

    # -------- Build response (unchanged) --------
    # First 5 rows
    head_df = ohlc.head(5).copy()
    head_rows = [
        {
            "date": idx.strftime("%Y-%m-%d"),
            **{col.lower(): (None if pd.isna(val) else float(val)) for col, val in row.items()}
        }
        for idx, row in head_df.iterrows()
    ]

    # Moving average bundles
    ma = moving_average_frames(ohlc)

    def _series(df, price_col):
        dates = [d.strftime("%Y-%m-%d") for d in df.index.to_pydatetime()]
        out = {
            "dates": dates,
            "price": [None if pd.isna(v) else float(v) for v in df[price_col].tolist()]
        }
        if "MA30" in df.columns:
            out["ma30"] = [None if pd.isna(v) else float(v) for v in df["MA30"].tolist()]
        if "MA50" in df.columns:
            out["ma50"] = [None if pd.isna(v) else float(v) for v in df["MA50"].tolist()]
        return out

    resp = {
        "ticker": ticker,
        "first_5_rows": head_rows,
        "charts": {
            "close_ma": _series(ma.get("close_ma", ohlc[["Close"]].rename(columns={"Close":"Close"})), "Close"),
            "open_ma":  _series(ma.get("open_ma",  ohlc[["Open"]].rename(columns={"Open":"Open"})),   "Open"),
            "high_ma":  _series(ma.get("high_ma",  ohlc[["High"]].rename(columns={"High":"High"})),   "High"),
            "close_raw": {
                "dates": [d.strftime("%Y-%m-%d") for d in ohlc.index.to_pydatetime()],
                "price": [float(x) for x in ohlc["Close"].tolist()]
            },
            "open_raw": {
                "dates": [d.strftime("%Y-%m-%d") for d in ohlc.index.to_pydatetime()],
                "price": [float(x) for x in ohlc["Open"].tolist()]
            }
        }
    }
    return jsonify(resp)

    # except Exception as e:
    #     log.exception("Analysis fetch failed")
    #     return jsonify({"error": f"Analysis failed: {e}"}), 500



