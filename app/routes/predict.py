# app/routes/predict.py
from __future__ import annotations

from flask import Blueprint, request, jsonify
import logging

from app.services.validators import validate_predict_payload, ValidationError
from app.services.model_manager import ModelManager
from app.services.forecaster import validate_on_price, forecast_n_days

bp = Blueprint("predict", __name__)
log = logging.getLogger(__name__)

# Reuse a single manager per worker
_model_manager = ModelManager()

_SERIES_KEY = {
    "close": "pred_close",
    "open":  "pred_open",
    "high":  "pred_high",
}

@bp.post("/predict")
def predict():
    """POST /api/predict
    JSON body:
      {
        "head": "close" | "open" | "high",
        "ticker": "AAPL" | "MSFT" | ...,
        "days": 1..MAX_FUTURE_DAYS
      }
    Returns JSON with dates + predicted series (+ optional metrics).
    """
    try:
        payload = request.get_json(force=True, silent=False)
        head, ticker, days = validate_predict_payload(payload)
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        log.exception("Invalid request payload")
        return jsonify({"error": "Invalid JSON payload."}), 400

    try:
        # Prepare model/scaler/sequences (AAPL → base, others → ephemeral finetune)
        run = _model_manager.prepare_for_ticker(head=head, ticker=ticker)

        # Optional: quick validation on price scale (val split)
        metrics = validate_on_price(run)

        # Forecast N future trading days
        fc = forecast_n_days(mm=_model_manager, run=run, n_future=days)

        series_key = _SERIES_KEY[head]
        dates = [d.strftime("%Y-%m-%d") for d in fc.index.to_pydatetime()]
        preds = [float(x) for x in fc[series_key].tolist()]

        # We keep the response minimal for the UI: graph + table only
        return jsonify({
            "head": head,
            "ticker": ticker,
            "was_finetuned": bool(run.was_finetuned),
            "dates": dates,
            "series_name": series_key,
            "predicted": preds,
            "metrics": metrics,   # MAE/RMSE/MAPE/R2/DA (for display if you want)
        })

    except Exception as e:
        log.exception("Prediction failed")
        return jsonify({"error": f"Prediction failed: {e}"}), 500
