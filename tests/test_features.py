import numpy as np
import pytest

import features as ft

DAY0 = np.datetime64("2026-01-01")
N_DAYS = 40
LAST = 29


def make_state(cust_daily: np.ndarray, term_n: np.ndarray, term_fraud: np.ndarray) -> ft.State:
    """One customer "c" and terminal "t"; one array of daily values (N_DAYS entries)."""
    days = np.arange(N_DAYS)
    customers = ft.DailyCounts.from_table(
        np.full(N_DAYS, "c"), days, {"n": cust_daily, "amount_sum": cust_daily * 10.0}, N_DAYS
    )
    terminals = ft.DailyCounts.from_table(
        np.full(N_DAYS, "t"), days, {"n": term_n, "n_fraud": term_fraud}, N_DAYS
    )
    return ft.State(DAY0, LAST, customers, terminals)


def features_on(state: ft.State, day: int, customer="c", terminal="t", amount=20.0) -> dict:
    ts = np.array([DAY0 + np.timedelta64(day, "D")], dtype="datetime64[s]") + np.timedelta64(
        3600 * 13, "s"
    )
    X = ft.compute(state, ts, [customer], [terminal], np.array([amount]))
    return dict(zip(ft.feature_names(), X[0].tolist(), strict=True))


@pytest.fixture
def state():
    rng = np.random.default_rng(0)
    n = rng.integers(1, 9, N_DAYS).astype(float)
    return make_state(n, n * 3, rng.integers(0, 3, N_DAYS).astype(float))


def test_names_match_matrix_width(state):
    assert len(features_on(state, 20)) == len(ft.feature_names()) == 20


def test_raw_features():
    f = features_on(make_state(np.ones(N_DAYS), np.ones(N_DAYS), np.zeros(N_DAYS)), 2)
    assert f["hour"] == 13
    assert f["weekday"] == 5 and f["is_weekend"] == 1  # 2026-01-03 is a Saturday
    assert f["log_amount"] == pytest.approx(np.log1p(20.0))


def test_windows_read_only_previous_days():
    n = np.zeros(N_DAYS)
    n[[20, 19, 13, 6]] = [100, 1, 2, 4]
    f = features_on(make_state(n, n, np.zeros(N_DAYS)), 20)
    assert (f["cust_n_1d"], f["cust_n_7d"], f["cust_n_14d"]) == (1, 3, 7)
    assert f["cust_amount_mean_7d"] == pytest.approx(10.0)
    assert f["amount_ratio_7d"] == pytest.approx(2.0)
    assert f["term_n_14d"] == 7


def test_fraud_rate_waits_for_the_label_delay():
    n, fraud = np.full(N_DAYS, 10.0), np.zeros(N_DAYS)
    fraud[13:20] = 10  # fraud in the previous 7 days: labels are not yet available
    f = features_on(make_state(n, n, fraud), 20)
    assert f["term_fraud_rate_1d"] == f["term_fraud_rate_14d"] == 0
    fraud[12] = 5  # D-8: the most recent day whose label is available
    f = features_on(make_state(n, n, fraud), 20)
    assert f["term_fraud_rate_1d"] == pytest.approx(0.5)
    assert f["term_fraud_rate_7d"] == pytest.approx(5 / 70)


@pytest.mark.parametrize("day", [15, 20, 29])
def test_no_feature_depends_on_the_future(state, day):
    expected = features_on(state, day)
    rng = np.random.default_rng(1)
    for counts, col, first_hidden in (
        (state.customers, "n", day),
        (state.customers, "amount_sum", day),
        (state.terminals, "n", day),
        (state.terminals, "n_fraud", day - ft.LABEL_DELAY_DAYS),
    ):
        bump = np.zeros(N_DAYS + 1)
        bump[first_hidden + 1 :] = np.cumsum(rng.integers(1, 50, N_DAYS - first_hidden))
        counts.cum[col][0] += bump
    assert features_on(state, day) == pytest.approx(expected, nan_ok=True)


def test_unknown_ids_give_missing_values(state):
    f = features_on(state, 20, customer="?", terminal="?")
    assert f["cust_n_14d"] == 0 and f["term_n_14d"] == 0
    assert np.isnan(f["cust_amount_mean_14d"]) and np.isnan(f["amount_ratio_14d"])
    assert np.isnan(f["term_fraud_rate_14d"])


def test_transaction_after_last_known_day_reads_the_last_windows(state):
    calendar = ("weekday", "is_weekend")
    late, reference = features_on(state, LAST + 5), features_on(state, LAST + 1)
    assert {k: v for k, v in late.items() if k not in calendar} == pytest.approx(
        {k: v for k, v in reference.items() if k not in calendar}, nan_ok=True
    )


def test_stream_updates_equal_a_state_rebuilt_from_tables():
    rng = np.random.default_rng(2)
    n = rng.integers(1, 9, N_DAYS).astype(float)
    n[LAST + 1 :] = 0
    fraud = np.zeros(N_DAYS)
    streamed = make_state(n, n, fraud)
    for _ in range(4):
        streamed.add_transaction(DAY0 + np.timedelta64(LAST + 1, "D"), "c", "t", 10.0)
    rebuilt_n = n.copy()
    rebuilt_n[LAST + 1] = 4
    rebuilt = make_state(rebuilt_n, rebuilt_n, fraud)
    rebuilt.last_day = LAST + 1
    assert streamed.last_day == LAST + 1
    assert features_on(streamed, LAST + 2) == pytest.approx(features_on(rebuilt, LAST + 2))
    assert features_on(streamed, LAST + 1) == pytest.approx(features_on(rebuilt, LAST + 1))
