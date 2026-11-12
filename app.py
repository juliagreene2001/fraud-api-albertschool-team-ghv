"""Service de scoring : GET /health, POST /predict, POST /events, POST /pubsub.

Le modèle et les compteurs journaliers sont chargés une fois par worker, au démarrage.
"""

from __future__ import annotations

import base64
import json
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from time import perf_counter

import numpy as np
import psutil
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError

import features as ft
import scoring
import settings


class Transaction(BaseModel):
    """Une transaction FraudFinder ; les noms en majuscules du flux (TX_ID...) sont acceptés."""

    model_config = ConfigDict(alias_generator=str.upper, populate_by_name=True)

    tx_id: str
    tx_ts: datetime
    customer_id: str
    terminal_id: str
    tx_amount: float = Field(gt=0)


class PredictRequest(BaseModel):
    transactions: list[Transaction]


class ScoredTransaction(BaseModel):
    tx_id: str
    score: float


class PredictResponse(BaseModel):
    scores: list[ScoredTransaction]
    timings_ms: dict[str, float]
    worker_pid: int


class PushMessage(BaseModel):
    data: str
    messageId: str
    publishTime: str | None = None
    attributes: dict[str, str] = Field(default_factory=dict)


class PushEnvelope(BaseModel):
    message: PushMessage
    subscription: str
    deliveryAttempt: int | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = settings.from_env()
    app.state.cfg = cfg
    app.state.model = scoring.load_model(cfg.models_dir, cfg.xgb_nthread)
    app.state.features = ft.load_state(cfg.data_dir, cfg.horizon_days)
    yield


app = FastAPI(title="TD2 fraud API", lifespan=lifespan)


def _utc_naive(ts: datetime) -> datetime:
    return ts.astimezone(UTC).replace(tzinfo=None) if ts.tzinfo else ts


def _arrays(transactions: list[Transaction]):
    ts = np.array([_utc_naive(t.tx_ts) for t in transactions], dtype="datetime64[s]")
    amount = np.array([t.tx_amount for t in transactions], dtype=np.float64)
    return (
        ts,
        [t.customer_id for t in transactions],
        [t.terminal_id for t in transactions],
        amount,
    )


def _predict(req: PredictRequest, request: Request, record: bool) -> PredictResponse:
    start = perf_counter()
    st = request.app.state
    ts, customers, terminals, amount = _arrays(req.transactions)
    scores, tm = scoring.score(st.features, st.model, ts, customers, terminals, amount)
    if record:
        for t, c, term, a in zip(ts, customers, terminals, amount, strict=True):
            st.features.add_transaction(t, c, term, float(a))
    return PredictResponse(
        scores=[
            ScoredTransaction(tx_id=t.tx_id, score=float(s))
            for t, s in zip(req.transactions, scores, strict=True)
        ],
        timings_ms={
            "features": tm.features * 1000,
            "predict": tm.predict * 1000,
            "handler": (perf_counter() - start) * 1000,
        },
        worker_pid=os.getpid(),
    )


@app.get("/health")
def health(request: Request):
    state = request.app.state.features
    return {
        "status": "ok",
        "pid": os.getpid(),
        "features": len(ft.feature_names()),
        "customers": len(state.customers.index),
        "terminals": len(state.terminals.index),
        "last_day": str(state.day0 + np.timedelta64(state.last_day, "D")),
        "rss_mb": round(psutil.Process().memory_info().rss / 2**20, 1),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request) -> PredictResponse:
    """Score sans rien mémoriser : deux appels identiques donnent la même réponse."""
    return _predict(req, request, record=False)


@app.post("/events", response_model=PredictResponse)
def events(req: PredictRequest, request: Request) -> PredictResponse:
    """Score puis enregistre les transactions dans les compteurs du worker (flux)."""
    return _predict(req, request, record=True)


def _log(severity: str, message: str, **fields) -> None:
    # Une ligne JSON sur stdout : Cloud Logging la lit comme jsonPayload.
    print(json.dumps({"severity": severity, "message": message, **fields}), flush=True)


@app.post("/pubsub", status_code=204)
def pubsub(envelope: PushEnvelope, request: Request) -> Response:
    """Message Pub/Sub en push : score sans rien mémoriser, puis acquitte par un 204.

    Tout code autre que 2xx fait rejouer le message, puis le mène à la file des messages morts.
    """
    ids = {
        "message_id": envelope.message.messageId,
        "delivery_attempt": envelope.deliveryAttempt,
    }
    try:
        payload = json.loads(base64.b64decode(envelope.message.data, validate=True))
    except ValueError:
        _log("ERROR", "rejected", reason="malformed", **ids)
        return Response(status_code=400)
    try:
        tx = Transaction.model_validate(payload)
    except ValidationError:
        tx_id = payload.get("TX_ID") if isinstance(payload, dict) else None
        _log("ERROR", "rejected", tx_id=tx_id, reason="invalid", **ids)
        return Response(status_code=422)
    score = (
        _predict(PredictRequest(transactions=[tx]), request, record=False)
        .scores[0]
        .score
    )
    _log("INFO", "scored", tx_id=tx.tx_id, score=score, **ids)
    return Response(status_code=204)
