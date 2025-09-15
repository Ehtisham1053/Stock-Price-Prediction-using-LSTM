from flask import Blueprint, jsonify, current_app
import os

bp = Blueprint("health", __name__)

@bp.get("/health")
def health():
    cfg = current_app.config
    bundles = {
        "close": os.path.isdir(cfg["BASE_AAPL_CLOSE"]),
        "open":  os.path.isdir(cfg["BASE_AAPL_OPEN"]),
        "high":  os.path.isdir(cfg["BASE_AAPL_HIGH"]),
    }
    return jsonify({
        "status": "ok",
        "lookback": cfg["LOOKBACK"],
        "calendar": cfg["CALENDAR"],
        "bundles_present": bundles
    })
