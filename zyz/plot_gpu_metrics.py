import sys
import csv
from pathlib import Path

import httpx

MAGNUS_ADDRESS = "http://162.105.151.134:3011/"
MAGNUS_TOKEN = "sk-xxx"
OUTPUT_DIR = Path(__file__).resolve().parent / "image"

API_BASE = MAGNUS_ADDRESS.rstrip("/") + "/api"
HEADERS = {"Authorization": f"Bearer {MAGNUS_TOKEN}"}

METRICS = [
    ("system.gpu.memory.used_bytes", "gpu_memory_used_bytes.csv"),
    ("system.gpu.utilization", "gpu_utilization.csv"),
]


def fetch_points(job_id: str, metric_name: str):
    url = f"{API_BASE}/jobs/{job_id}/metrics/query"
    params = {"name": metric_name, "max_points": 10000}
    resp = httpx.get(url, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json().get("points", [])


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_gpu_metrics.py <job_id>")
        sys.exit(1)

    job_id = sys.argv[1]

    for metric_name, filename in METRICS:
        output_path = OUTPUT_DIR / filename
        print(f"Fetching {metric_name} ...")
        points = fetch_points(job_id, metric_name)
        if not points:
            print(f"  No data for {metric_name}")
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_unix_ms", "step", "value", "device", "node"])
            for p in points:
                labels = p.get("labels") or {}
                writer.writerow([
                    p.get("time_unix_ms", ""),
                    p.get("step", ""),
                    p.get("value", ""),
                    labels.get("device", ""),
                    labels.get("node", ""),
                ])
        print(f"  Saved {len(points)} rows to {output_path}")


if __name__ == "__main__":
    main()
