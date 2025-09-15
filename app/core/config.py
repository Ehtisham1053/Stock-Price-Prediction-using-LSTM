import os

class Config:
    # Core model pipeline params (must match training)
    LOOKBACK = int(os.getenv("LOOKBACK", "60"))
    TRIM_YEARS = int(os.getenv("TRIM_YEARS", "10"))
    MAX_FUTURE_DAYS = int(os.getenv("MAX_FUTURE_DAYS", "252"))
    CALENDAR = os.getenv("CALENDAR", "XNYS")

    ANALYSIS_YEARS = int(os.getenv("ANALYSIS_YEARS", "10"))

    # Model bundle paths (AAPL bases already saved here)
    MODELS_DIR = os.getenv("MODELS_DIR", "models")
    BASE_AAPL_CLOSE = os.path.join(MODELS_DIR, "base_aapl")
    BASE_AAPL_OPEN  = os.path.join(MODELS_DIR, "base_aapl_open")
    BASE_AAPL_HIGH  = os.path.join(MODELS_DIR, "base_aapl_high")

    # Server / logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


