import pytest
from fastapi.testclient import TestClient

from tests.conftest import needs_data

pytestmark = needs_data


@pytest.fixture
def client():
    import app

    with TestClient(app.app) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["features"] == 20
    assert body["customers"] > 40_000 and body["terminals"] > 4_900
    assert body["last_day"] == "2026-09-30"


def test_predict_matches_reference_scores(client, replay, golden):
    body = client.post("/predict", json={"transactions": replay}).json()
    assert [s["tx_id"] for s in body["scores"]] == golden["tx_id"]
    assert [s["score"] for s in body["scores"]] == pytest.approx(golden["scores"], abs=1e-6)
    assert set(body["timings_ms"]) == {"features", "predict", "handler"}


def test_one_by_one_equals_batch(client, replay, golden):
    for tx, expected in list(zip(replay, golden["scores"], strict=True))[:20]:
        score = client.post("/predict", json={"transactions": [tx]}).json()["scores"][0]["score"]
        assert score == pytest.approx(expected, abs=1e-6)


def test_fraudfinder_message_format_is_accepted(client, replay, golden):
    tx = replay[0]
    message = {
        "TX_ID": tx["tx_id"],
        "TX_TS": tx["tx_ts"][:19].replace("T", " "),
        "CUSTOMER_ID": tx["customer_id"],
        "TERMINAL_ID": tx["terminal_id"],
        "TX_AMOUNT": tx["tx_amount"],
    }
    score = client.post("/predict", json={"transactions": [message]}).json()["scores"][0]["score"]
    assert score == pytest.approx(golden["scores"][0], abs=1e-6)


def test_predict_does_not_change_the_state(client, replay):
    first = client.post("/predict", json={"transactions": replay}).json()["scores"]
    assert client.post("/predict", json={"transactions": replay}).json()["scores"] == first


def test_empty_batch(client):
    assert client.post("/predict", json={"transactions": []}).json()["scores"] == []


@pytest.mark.parametrize("patch", [{"tx_amount": 0}, {"tx_ts": "yesterday"}, {"customer_id": None}])
def test_invalid_transaction_is_rejected(client, replay, patch):
    assert client.post("/predict", json={"transactions": [replay[0] | patch]}).status_code == 422
