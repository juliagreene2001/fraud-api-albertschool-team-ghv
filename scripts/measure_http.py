"""Measure client-observed HTTP latency; preserve failures and raw samples in JSON."""

from __future__ import annotations

import argparse
import json
import math
import ssl
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import certifi

ROOT = Path(__file__).resolve().parents[1]
TLS = ssl.create_default_context(cafile=certifi.where())


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--endpoint", choices=["/health", "/predict"], default="/health")
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--payload", type=Path, default=ROOT / "examples/transaction.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1:
        parser.error("Request count and concurrency must be positive")
    headers = {}
    if args.private:
        token = subprocess.check_output(
            ["gcloud", "auth", "print-identity-token"], text=True
        ).strip()
        headers["Authorization"] = f"Bearer {token}"
    body = None
    if args.endpoint == "/predict":
        body = args.payload.read_bytes()
        json.loads(body)
        headers["Content-Type"] = "application/json"
    url = args.url.rstrip("/") + args.endpoint

    def send(index: int) -> dict:
        start = perf_counter()
        try:
            request = Request(url, data=body, headers=headers)
            with urlopen(request, timeout=120, context=TLS) as response:
                result = json.load(response)
                sample = {"index": index, "status": response.status}
                if "timings_ms" in result:
                    sample["server_timings_ms"] = result["timings_ms"]
                sample["worker_pid"] = result.get("worker_pid", result.get("pid"))
        except (HTTPError, URLError, TimeoutError, ValueError) as error:
            sample = {"index": index, "status": getattr(error, "code", None)}
            sample["error"] = str(error)
        sample["latency_ms"] = (perf_counter() - start) * 1000
        return sample

    started = datetime.now(UTC).isoformat()
    clock = perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        samples = list(pool.map(send, range(args.requests)))
    elapsed = perf_counter() - clock
    successes = [sample["latency_ms"] for sample in samples if sample["status"] == 200]
    report = {
        "measurement": "client HTTP round-trip, including connection/network overhead",
        "url": url,
        "started_utc": started,
        "finished_utc": datetime.now(UTC).isoformat(),
        "requests": args.requests,
        "concurrency": args.concurrency,
        "successes": len(successes),
        "failures": args.requests - len(successes),
        "elapsed_seconds": elapsed,
        "p50_ms": percentile(successes, 0.50) if successes else None,
        "p99_ms": percentile(successes, 0.99) if successes else None,
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "samples"}, indent=2))
    if len(successes) != args.requests:
        raise SystemExit("Some requests failed; see raw samples in the output file")


if __name__ == "__main__":
    main()
