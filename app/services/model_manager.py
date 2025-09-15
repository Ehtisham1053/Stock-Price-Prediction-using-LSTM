# app/services/model_manager.py
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional, Literal

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers, losses, metrics, backend as K

from app.core.config import Config
from app.services.sequences import (
    prepare_sequences_for_head,  # must accept: ticker, head, use_base_scaler, lookback, trim_years, calendar
    PrepResult,
)

# Type alias for heads
Head = Literal["close", "open", "high"]

logger = logging.getLogger(__name__)

# -----------------------------
# Keras custom objects
# -----------------------------
def r2_metric(y_true, y_pred):
    # Safe R^2 (avoid NaNs)
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1.0 - K.switch(K.equal(ss_tot, 0.0), 0.0, ss_res / (ss_tot + K.epsilon()))

CUSTOM_OBJECTS = {"r2_metric": r2_metric}

# -----------------------------
# Dataclasses
# -----------------------------
@dataclass(frozen=True)
class BaseBundle:
    model: keras.Model           # immutable base (do NOT train this instance)
    scaler: StandardScaler
    cfg: dict

@dataclass
class ManagerRun:
    head: Head
    ticker: str
    model: keras.Model           # ready for predict() (AAPL) or already fine-tuned (non-AAPL)
    scaler: StandardScaler       # scaler used for this run (base for AAPL; train-only for others)
    prep: PrepResult             # sequences, frames, etc., produced with the correct lookback
    was_finetuned: bool
    history: Optional[dict]      # training history if finetuned; else None


# -----------------------------
# ModelManager
# -----------------------------
class ModelManager:
    """
    Loads cached AAPL base bundles (close/open/high), and prepares a model+scaler+data
    for any ticker. For non-AAPL, performs *ephemeral* fine-tuning in memory on a
    CLONED model (base cache remains untouched).
    """

    def __init__(
        self,
        *,
        ft_epochs: int = 60,
        ft_batch_size: int = 64,
        ft_lr: float = 1e-4,
    ) -> None:
        self.ft_epochs = ft_epochs
        self.ft_batch_size = ft_batch_size
        self.ft_lr = ft_lr

        # caches for base bundles
        self._base_cache: Dict[Head, BaseBundle] = {}

        # resolve base paths from Config (must point to model.keras & scaler.pkl)
        # Your Config already defines these:
        #   BASE_AAPL_CLOSE, BASE_AAPL_OPEN, BASE_AAPL_HIGH
        self._base_paths: Dict[Head, str] = {
            "close": Config.BASE_AAPL_CLOSE,
            "open":  Config.BASE_AAPL_OPEN,
            "high":  Config.BASE_AAPL_HIGH,
        }

    # ---------- public API ----------

    def prepare_for_ticker(self, head: Head, ticker: str) -> ManagerRun:
        """
        AAPL:
          - Load base bundle once (cached)
          - Infer lookback from base.model.input_shape[1]
          - Prepare sequences using *base scaler* and that lookback
          - Return base model (no compile needed to predict)

        Non-AAPL:
          - Load base bundle (cached), infer lookback
          - Prepare sequences with TRAIN-fitted scaler (use_base_scaler=None)
          - Clone base model (copy weights), compile, fine-tune in RAM
          - Return finetuned clone (NOT saved), plus new scaler
        """
        head = _norm_head(head)
        ticker = ticker.upper().strip()

        base = self._get_base_bundle(head)
        expected_lb = _infer_lookback_from_model(base.model)

        if ticker == "AAPL":
            # Prep with the *base scaler* to mirror notebook behavior
            prep = prepare_sequences_for_head(
                ticker=ticker,
                head=head,
                use_base_scaler=base.scaler,   # transform with the same scaler used at train time
                lookback=expected_lb,          # CRITICAL: match model input shape
                trim_years=Config.TRIM_YEARS,
                calendar=Config.CALENDAR,
            )
            _assert_seq_shape(prep.seq.X_val, expected_lb, context=f"{head}/AAPL X_val")
            # Predict path doesn't need compile; keras predict() works without compile.
            return ManagerRun(
                head=head, ticker=ticker,
                model=base.model, scaler=base.scaler,
                prep=prep, was_finetuned=False, history=None
            )

        # Non-AAPL: fit scaler on TRAIN only (prep does that when use_base_scaler=None)
        prep = prepare_sequences_for_head(
            ticker=ticker,
            head=head,
            use_base_scaler=None,             # fit fresh scaler on this ticker's TRAIN
            lookback=expected_lb,             # CRITICAL: match model input shape
            trim_years=Config.TRIM_YEARS,
            calendar=Config.CALENDAR,
        )
        _assert_seq_shape(prep.seq.X_train, expected_lb, context=f"{head}/{ticker} X_train")
        _assert_seq_shape(prep.seq.X_val,   expected_lb, context=f"{head}/{ticker} X_val")

        # CLONE the cached base model so the cache remains immutable
        model = _clone_model_with_weights(base.model)
        self._compile_for_finetune(model)
        hist = self._train_in_memory(
            model=model,
            X_train=prep.seq.X_train, y_train=prep.seq.y_train,
            X_val=prep.seq.X_val,     y_val=prep.seq.y_val
        )

        return ManagerRun(
            head=head, ticker=ticker,
            model=model, scaler=prep.scaled.scaler,
            prep=prep, was_finetuned=True, history=hist
        )

    # ---------- internals ----------

    def _get_base_bundle(self, head: Head) -> BaseBundle:
        if head in self._base_cache:
            return self._base_cache[head]

        base_dir = self._base_paths[head]
        model_path  = os.path.join(base_dir, "model.keras")
        scaler_path = os.path.join(base_dir, "scaler.pkl")
        cfg_path    = os.path.join(base_dir, "config.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[{head}] Missing model file: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"[{head}] Missing scaler file: {scaler_path}")

        logger.info("Loading base bundle for %s from %s", head, base_dir)
        model = keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS, compile=False)
        scaler = joblib.load(scaler_path)
        cfg = {}
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r") as f:
                    cfg = json.load(f)
            except Exception:
                logger.warning("Could not parse config.json at %s", cfg_path)

        # Sanity: ensure the model indeed expects (None, lookback, 1)
        lb = _infer_lookback_from_model(model)
        if lb <= 0:
            raise ValueError(f"[{head}] Invalid lookback inferred from model.input_shape: {model.input_shape}")

        bundle = BaseBundle(model=model, scaler=scaler, cfg=cfg)
        self._base_cache[head] = bundle
        return bundle

    def _compile_for_finetune(self, model: keras.Model) -> None:
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.ft_lr),
            loss=losses.Huber(delta=1.0),
            metrics=[metrics.MAE, metrics.RootMeanSquaredError(name="rmse")]
        )

    def _train_in_memory(
        self,
        *,
        model: keras.Model,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val:   np.ndarray, y_val:   np.ndarray,
    ) -> dict:
        cbs = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=12, restore_best_weights=True, verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=6,
                min_delta=1e-4, min_lr=1e-6, verbose=1
            ),
            keras.callbacks.TerminateOnNaN(),
        ]
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.ft_epochs,
            batch_size=self.ft_batch_size,
            shuffle=False,
            verbose=1,
            callbacks=cbs,
        )
        # lightweight, JSON-safe
        return {k: [float(x) for x in v] for k, v in history.history.items()}


# -----------------------------
# helpers
# -----------------------------
def _norm_head(head: str) -> Head:
    h = head.lower().strip()
    if h not in {"close", "open", "high"}:
        raise ValueError("head must be one of: close | open | high")
    return h  # type: ignore

def _infer_lookback_from_model(model: keras.Model) -> int:
    """
    Extract lookback from model.input_shape (expects (None, LB, C)).
    Raises if incompatible.
    """
    ishape = getattr(model, "input_shape", None)
    if not ishape or len(ishape) < 3:
        raise ValueError(f"Model has unexpected input_shape: {ishape}")
    lb = int(ishape[1])
    ch = int(ishape[2])
    if ch != 1:
        raise ValueError(f"Model expects {ch} channels; pipeline assumes 1 feature.")
    return lb

def _assert_seq_shape(x: np.ndarray, lookback: int, *, context: str) -> None:
    """
    Ensure sequences are (N, lookback, 1). Raise with a clear message if not.
    """
    if x.ndim != 3 or x.shape[1] != lookback or x.shape[2] != 1:
        raise ValueError(
            f"[{context}] Sequence shape mismatch: expected (N,{lookback},1), got {x.shape}. "
            f"Check that prepare_sequences_for_head() received lookback={lookback} and is reshaping with a channel dim."
        )

def _clone_model_with_weights(base: keras.Model) -> keras.Model:
    """
    Clone a compiled/compiled=False model and copy weights so the cached base remains immutable.
    """
    clone = keras.models.clone_model(base)
    clone.set_weights(base.get_weights())
    # leave uncompiled; caller decides whether to compile (we compile only for finetune)
    return clone
