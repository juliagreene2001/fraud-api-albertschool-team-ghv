"""Scoring features computed at the transaction date from daily counters.

The same implementation supports training (millions of rows) and serving (one row).

Availability rules for a transaction on day D:
- customer and terminal counters: days D-w through D-1 (day D itself is never read);
- terminal fraud rate: days D-w-LABEL_DELAY_DAYS through D-1-LABEL_DELAY_DAYS, because a
  transaction label becomes available only LABEL_DELAY_DAYS days after the transaction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

WINDOWS = (1, 7, 14)
LABEL_DELAY_DAYS = 7
RAW_FEATURES = ("amount", "log_amount", "hour", "weekday", "is_weekend")
CUSTOMER_FILE = "customer_daily.parquet"
TERMINAL_FILE = "terminal_daily.parquet"


def feature_names() -> list[str]:
    names = list(RAW_FEATURES)
    for w in WINDOWS:
        names += [f"cust_n_{w}d", f"cust_amount_mean_{w}d", f"amount_ratio_{w}d"]
    for w in WINDOWS:
        names += [f"term_n_{w}d", f"term_fraud_rate_{w}d"]
    return names


class DailyCounts:
    """Daily counters per entity, stored as cumulative sums: cum[col][e, j] is the total
    for entity e over days 0 through j-1. A window is the difference between two entries."""

    def __init__(self, ids: np.ndarray, n_days: int, columns: tuple[str, ...]):
        self.index = {v: i for i, v in enumerate(ids.tolist())}
        self.n_days = n_days
        self.cum = {c: np.zeros((len(ids) + 1, n_days + 1), dtype=np.float64) for c in columns}

    @classmethod
    def from_table(
        cls, ids: np.ndarray, day_idx: np.ndarray, values: dict[str, np.ndarray], n_days: int
    ) -> DailyCounts:
        uniq, rows = np.unique(ids, return_inverse=True)
        out = cls(uniq, n_days, tuple(values))
        for col, v in values.items():
            daily = np.zeros((len(uniq) + 1, n_days), dtype=np.float64)
            np.add.at(daily, (rows, day_idx), v)
            out.cum[col][:, 1:] = np.cumsum(daily, axis=1)
        return out

    def rows(self, ids) -> np.ndarray:
        """Row for each ID; unknown IDs map to the final row, which is always empty."""
        unknown = len(self.index)
        return np.fromiter(
            (self.index.get(i, unknown) for i in ids), dtype=np.int64, count=len(ids)
        )

    def window(self, col: str, rows: np.ndarray, end: np.ndarray, w: int) -> np.ndarray:
        """Total for the w days preceding day `end` (exclusive)."""
        hi = np.clip(end, 0, self.n_days)
        lo = np.clip(hi - w, 0, self.n_days)
        c = self.cum[col]
        return c[rows, hi] - c[rows, lo]

    def add(self, col: str, entity: str, day: int, value: float) -> None:
        row = self.index.get(entity)
        if row is not None and 0 <= day < self.n_days:
            self.cum[col][row, day + 1 :] += value


@dataclass
class State:
    day0: np.datetime64
    last_day: int
    customers: DailyCounts
    terminals: DailyCounts

    def day_index(self, ts: np.ndarray) -> np.ndarray:
        return (ts.astype("datetime64[D]") - self.day0).astype(np.int64)

    def add_transaction(self, ts: np.datetime64, customer_id: str, terminal_id: str, amount: float):
        """Record a stream transaction; it contributes to windows starting the next day."""
        day = int(self.day_index(np.asarray(ts)))
        self.customers.add("n", customer_id, day, 1.0)
        self.customers.add("amount_sum", customer_id, day, amount)
        self.terminals.add("n", terminal_id, day, 1.0)
        if day < self.customers.n_days:
            self.last_day = max(self.last_day, day)


def _days(table, day0: np.datetime64) -> np.ndarray:
    return (table["day"].to_numpy().astype("datetime64[D]") - day0).astype(np.int64)


def load_state(data_dir: Path, horizon_days: int = 0) -> State:
    """Load daily counters; `horizon_days` reserves empty days after the last known day
    for incoming stream transactions."""
    cust = pq.read_table(data_dir / CUSTOMER_FILE)
    term = pq.read_table(data_dir / TERMINAL_FILE)
    first = min(cust["day"].to_numpy().min(), term["day"].to_numpy().min()).astype("datetime64[D]")
    last = max(cust["day"].to_numpy().max(), term["day"].to_numpy().max()).astype("datetime64[D]")
    last_day = int((last - first).astype(np.int64))
    n_days = last_day + 1 + horizon_days
    customers = DailyCounts.from_table(
        cust["customer_id"].to_numpy(zero_copy_only=False),
        _days(cust, first),
        {"n": cust["n"].to_numpy(), "amount_sum": cust["amount_sum"].to_numpy()},
        n_days,
    )
    terminals = DailyCounts.from_table(
        term["terminal_id"].to_numpy(zero_copy_only=False),
        _days(term, first),
        {"n": term["n"].to_numpy(), "n_fraud": term["n_fraud"].to_numpy()},
        n_days,
    )
    return State(first, last_day, customers, terminals)


def compute(
    state: State, ts: np.ndarray, customer_ids, terminal_ids, amount: np.ndarray
) -> np.ndarray:
    """Return a (B, F) float32 matrix in feature_names() order.

    `ts` is a UTC datetime64 array. Transactions after the last known day read windows
    ending at that last known day.
    """
    ts = ts.astype("datetime64[s]")
    amount = amount.astype(np.float64)
    days = ts.astype("datetime64[D]")
    epoch_day = days.astype(np.int64)
    weekday = (epoch_day + 3) % 7
    cols: dict[str, np.ndarray] = {
        "amount": amount,
        "log_amount": np.log1p(amount),
        "hour": (ts - days).astype("timedelta64[h]").astype(np.int64),
        "weekday": weekday,
        "is_weekend": weekday >= 5,
    }
    end = np.minimum(state.day_index(ts), state.last_day + 1)
    cust = state.customers.rows(customer_ids)
    term = state.terminals.rows(terminal_ids)
    with np.errstate(divide="ignore", invalid="ignore"):
        for w in WINDOWS:
            n = state.customers.window("n", cust, end, w)
            mean = state.customers.window("amount_sum", cust, end, w) / n
            cols[f"cust_n_{w}d"] = n
            cols[f"cust_amount_mean_{w}d"] = mean
            cols[f"amount_ratio_{w}d"] = amount / mean
        for w in WINDOWS:
            cols[f"term_n_{w}d"] = state.terminals.window("n", term, end, w)
            labelled = end - LABEL_DELAY_DAYS
            cols[f"term_fraud_rate_{w}d"] = state.terminals.window(
                "n_fraud", term, labelled, w
            ) / state.terminals.window("n", term, labelled, w)
    X = np.column_stack([cols[name] for name in feature_names()]).astype(np.float32)
    X[~np.isfinite(X)] = np.nan
    return X
