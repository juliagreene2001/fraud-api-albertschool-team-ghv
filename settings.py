"""Runtime configuration read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    data_dir: Path = ROOT / "data"
    models_dir: Path = ROOT / "models"
    xgb_nthread: int = 1
    horizon_days: int = 10


def from_env() -> Settings:
    return Settings(
        data_dir=Path(os.environ.get("DATA_DIR", ROOT / "data")),
        models_dir=Path(os.environ.get("MODELS_DIR", ROOT / "models")),
        xgb_nthread=int(os.environ.get("XGB_NTHREAD", 1)),
        horizon_days=int(os.environ.get("HORIZON_DAYS", 10)),
    )
