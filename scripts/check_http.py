"""Verify live health and 500 reference predictions; optionally verify private invocation."""

from __future__ import annotations

import argparse
import json
import ssl
import subprocess
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import certifi
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
TLS = ssl.create_default_context(cafile=certifi.where())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    base = args.url.rstrip("/")
    headers = {}
    report = {"url": base}
    if args.private:
        try:
            with urlopen(base + "/health", timeout=30, context=TLS) as response:
                status = response.status
        except HTTPError as error:
            status = error.code
        if status not in (401, 403):
            raise RuntimeError(f"Expected unauthenticated rejection, received HTTP {status}")
        report["unauthenticated_status"] = status
        token = subprocess.check_output(
            ["gcloud", "auth", "print-identity-token"], text=True
        ).strip()
        headers["Authorization"] = f"Bearer {token}"

    with urlopen(Request(base + "/health", headers=headers), timeout=120, context=TLS) as response:
        report["health"] = json.load(response)
    rows = pq.read_table(ROOT / "data/replay.parquet").slice(0, 500).to_pylist()
    transactions = [
        {
            "tx_id": row["tx_id"],
            "tx_ts": row["tx_ts"].isoformat(),
            "customer_id": row["customer_id"],
            "terminal_id": row["terminal_id"],
            "tx_amount": row["tx_amount"],
        }
        for row in rows
    ]
    request = Request(
        base + "/predict",
        data=json.dumps({"transactions": transactions}).encode(),
        headers=headers | {"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=120, context=TLS) as response:
        result = json.load(response)
    golden = json.loads((ROOT / "models/golden.json").read_text())
    assert [item["tx_id"] for item in result["scores"]] == golden["tx_id"]
    errors = [
        abs(item["score"] - expected)
        for item, expected in zip(result["scores"], golden["scores"], strict=True)
    ]
    assert max(errors) <= 1e-6, f"Reference mismatch: {max(errors)}"
    report.update(
        reference_predictions=len(errors),
        max_absolute_error=max(errors),
        timings_ms=result["timings_ms"],
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
