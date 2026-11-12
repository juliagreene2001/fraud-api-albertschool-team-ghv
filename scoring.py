"""Shared scoring path: features -> prediction, usable by the API, training, and tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import xgboost as xgb

import features as ft

MODEL_FILE = "model.ubj"


@dataclass
class Timings:
    features: float
    predict: float


def load_model(models_dir: Path, nthread: int) -> xgb.Booster:
    booster = xgb.Booster()
    booster.load_model(str(models_dir / MODEL_FILE))
    booster.set_param({"nthread": nthread})
    return booster


def score(
    state: ft.State,
    model: xgb.Booster,
    ts: np.ndarray,
    customer_ids,
    terminal_ids,
    amount: np.ndarray,
) -> tuple[np.ndarray, Timings]:
    """Return scores in input order and per-stage execution times in seconds."""
    t0 = perf_counter()
    X = ft.compute(state, ts, customer_ids, terminal_ids, amount)
    t1 = perf_counter()
    scores = model.inplace_predict(X, missing=np.nan) if len(X) else np.empty(0, dtype=np.float32)
    return scores, Timings(features=t1 - t0, predict=perf_counter() - t1)
