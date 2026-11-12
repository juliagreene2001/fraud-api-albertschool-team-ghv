import base64
import json

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
    assert [s["score"] for s in body["scores"]] == pytest.approx(
        golden["scores"], abs=1e-6
    )
    assert set(body["timings_ms"]) == {"features", "predict", "handler"}


def test_one_by_one_equals_batch(client, replay, golden):
    for tx, expected in list(zip(replay, golden["scores"], strict=True))[:20]:
        score = client.post("/predict", json={"transactions": [tx]}).json()["scores"][
            0
        ]["score"]
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
    score = client.post("/predict", json={"transactions": [message]}).json()["scores"][
        0
    ]["score"]
    assert score == pytest.approx(golden["scores"][0], abs=1e-6)


def test_predict_does_not_change_the_state(client, replay):
    first = client.post("/predict", json={"transactions": replay}).json()["scores"]
    assert (
        client.post("/predict", json={"transactions": replay}).json()["scores"] == first
    )


def test_empty_batch(client):
    assert client.post("/predict", json={"transactions": []}).json()["scores"] == []


@pytest.mark.parametrize(
    "patch", [{"tx_amount": 0}, {"tx_ts": "hier"}, {"customer_id": None}]
)
def test_invalid_transaction_is_rejected(client, replay, patch):
    assert (
        client.post("/predict", json={"transactions": [replay[0] | patch]}).status_code
        == 422
    )


def _envelope(data: str, attempt: int = 1) -> dict:
    return {
        "message": {"data": data, "messageId": "m-1", "attributes": {}},
        "subscription": "projects/advanced-cloud-ai/subscriptions/team-ghv-tx-push",
        "deliveryAttempt": attempt,
    }


def _encode(message: dict) -> str:
    return base64.b64encode(json.dumps(message).encode()).decode()


def _fraudfinder(tx: dict) -> dict:
    return {
        "TX_ID": tx["tx_id"],
        "TX_TS": tx["tx_ts"][:19].replace("T", " "),
        "CUSTOMER_ID": tx["customer_id"],
        "TERMINAL_ID": tx["terminal_id"],
        "TX_AMOUNT": tx["tx_amount"],
    }


def _logs(capsys) -> list[dict]:
    return [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("{")
    ]


def test_pubsub_scores_logs_and_acknowledges(client, replay, golden, capsys):
    response = client.post(
        "/pubsub", json=_envelope(_encode(_fraudfinder(replay[0])), attempt=2)
    )
    assert response.status_code == 204 and response.content == b""
    (line,) = _logs(capsys)
    assert line["severity"] == "INFO" and line["message"] == "scored"
    assert line["tx_id"] == replay[0]["tx_id"] and line["delivery_attempt"] == 2
    assert line["message_id"] == "m-1"
    assert line["score"] == pytest.approx(golden["scores"][0], abs=1e-6)


def test_pubsub_does_not_change_the_state(client, replay):
    before = client.post("/predict", json={"transactions": replay}).json()["scores"]
    for _ in range(2):
        assert (
            client.post(
                "/pubsub", json=_envelope(_encode(_fraudfinder(replay[0])))
            ).status_code
            == 204
        )
    assert (
        client.post("/predict", json={"transactions": replay}).json()["scores"]
        == before
    )


@pytest.mark.parametrize("amount", [0, -5])
def test_pubsub_rejects_non_positive_amount(client, replay, capsys, amount):
    message = _fraudfinder(replay[0]) | {"TX_AMOUNT": amount}
    assert client.post("/pubsub", json=_envelope(_encode(message))).status_code == 422
    (line,) = _logs(capsys)
    assert line["severity"] == "ERROR" and line["message"] == "rejected"
    assert line["tx_id"] == replay[0]["tx_id"]


@pytest.mark.parametrize(
    "data", ["%%not-base64%%", base64.b64encode(b"not json").decode(), _encode([1, 2])]
)
def test_pubsub_malformed_input_is_a_controlled_error(client, capsys, data):
    status = client.post("/pubsub", json=_envelope(data)).status_code
    assert status in (400, 422)
    assert _logs(capsys)[-1]["message"] == "rejected"
