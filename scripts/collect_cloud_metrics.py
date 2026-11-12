"""Read Cloud Monitoring samples and startup logs for one course service; never invent data."""

from __future__ import annotations

import argparse
import json
import ssl
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import certifi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", required=True)
    parser.add_argument("--revision")
    parser.add_argument(
        "--start", required=True, help="UTC ISO timestamp, for example 2026-11-12T10:00:00Z"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = "advanced-cloud-ai"
    token = subprocess.check_output(["gcloud", "auth", "print-access-token"], text=True).strip()
    tls = ssl.create_default_context(cafile=certifi.where())
    end = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    report = {
        "project": project,
        "service": args.service,
        "revision": args.revision,
        "start_utc": args.start,
        "end_utc": end,
        "source": "Google Cloud Monitoring API; samples can arrive after a reporting delay",
        "metrics": {},
    }

    def read_metric(metric: str, aligner: str | None = None) -> dict:
        scope = (
            f'metric.type="run.googleapis.com/{metric}" '
            f'AND resource.type="cloud_run_revision" '
            f'AND resource.labels.service_name="{args.service}"'
        )
        if args.revision:
            scope += f' AND resource.labels.revision_name="{args.revision}"'
        query = {
            "filter": scope,
            "interval.startTime": args.start,
            "interval.endTime": end,
            "view": "FULL",
            "pageSize": "1000",
        }
        if aligner:
            query["aggregation.alignmentPeriod"] = "300s"
            query["aggregation.perSeriesAligner"] = aligner
        url = f"https://monitoring.googleapis.com/v3/projects/{project}/timeSeries?"
        request = Request(url + urlencode(query), headers={"Authorization": f"Bearer {token}"})
        try:
            with urlopen(request, timeout=30, context=tls) as response:
                return json.load(response)
        except HTTPError as error:
            return {"http_error": error.code, "details": error.read().decode()}

    for metric in ("request_latencies", "container/startup_latencies", "container/instance_count"):
        report["metrics"][metric] = read_metric(metric)
    for metric in ("request_latencies", "container/startup_latencies"):
        for percentile in (50, 99):
            key = f"{metric}_p{percentile}_aligned_5min"
            report["metrics"][key] = read_metric(metric, f"ALIGN_PERCENTILE_{percentile}")

    counts = {}
    for series in report["metrics"]["container/instance_count"].get("timeSeries", []):
        for point in series.get("points", []):
            timestamp = point["interval"]["endTime"]
            value = point["value"]
            count = float(value.get("int64Value", value.get("doubleValue", 0)))
            counts[timestamp] = counts.get(timestamp, 0) + count
    report["observed_peak_instances"] = max(counts.values()) if counts else None
    report["instance_count_samples"] = counts

    log_filter = (
        'resource.type="cloud_run_revision" '
        f'AND resource.labels.service_name="{args.service}" '
        f'AND timestamp>="{args.start}" '
        'AND (textPayload:"Starting new instance" OR textPayload:"STARTUP" '
        'OR textPayload:"Application startup complete")'
    )
    if args.revision:
        log_filter += f' AND resource.labels.revision_name="{args.revision}"'
    logs = subprocess.run(
        [
            "gcloud", "logging", "read", log_filter, f"--project={project}", "--limit=200",
            "--format=json(timestamp,textPayload,resource.labels.revision_name)",
        ],
        capture_output=True,
        text=True,
    )
    report["startup_logs"] = json.loads(logs.stdout) if logs.returncode == 0 else logs.stderr
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "observed_peak_instances": report["observed_peak_instances"],
        "series_counts": {
            name: len(value.get("timeSeries", [])) for name, value in report["metrics"].items()
        },
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
