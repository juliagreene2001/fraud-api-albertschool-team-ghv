import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"

needs_data = pytest.mark.skipif(
    not all(
        path.exists()
        for path in (
            DATA / "customer_daily.parquet",
            DATA / "terminal_daily.parquet",
            DATA / "replay.parquet",
            MODELS / "model.ubj",
        )
    ),
    reason="Required history, replay, or model files are missing",
)


@pytest.fixture(scope="session")
def golden() -> dict:
    return json.loads((MODELS / "golden.json").read_text())


@pytest.fixture(scope="session")
def replay() -> list[dict]:
    """The first replay transactions, in the API's JSON format."""
    rows = pq.read_table(DATA / "replay.parquet").slice(0, 500).to_pylist()
    return [
        {
            "tx_id": r["tx_id"],
            "tx_ts": r["tx_ts"].isoformat(),
            "customer_id": r["customer_id"],
            "terminal_id": r["terminal_id"],
            "tx_amount": r["tx_amount"],
        }
        for r in rows
    ]
