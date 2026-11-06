#!/usr/bin/env python3
"""Run a small concurrent load test against the authenticated local Strata tunnel."""

from __future__ import annotations

import concurrent.futures
import datetime as dt
import getpass
import json
import statistics
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:18790"
MODEL = "qwen3.8-flash-next-iq3-xxs-strata-cann"
PROMPT = "The capital of France is"
MAX_TOKENS = 24
REPEATS_PER_WORKER = 3
CONCURRENCY_LEVELS = (1, 2, 4)
OUT = Path(__file__).with_name("strata-api-loadtest.json")


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * p
    lo, hi = int(pos), min(int(pos) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def main() -> None:
    api_key = getpass.getpass("Strata API key (input hidden): ").strip()
    if not api_key:
        raise SystemExit("API key is required")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    auth = {"Authorization": f"Bearer {api_key}"}

    def get_json(path: str) -> dict:
        req = urllib.request.Request(BASE + path, headers=auth)
        with opener.open(req, timeout=15) as response:
            return json.loads(response.read())

    def completion(n: int) -> dict:
        payload = json.dumps({
            "model": MODEL,
            "prompt": PROMPT,
            "max_tokens": n,
            "temperature": 0,
        }).encode()
        req = urllib.request.Request(
            BASE + "/v1/completions",
            data=payload,
            headers={**auth, "Content-Type": "application/json"},
        )
        started = time.perf_counter()
        try:
            with opener.open(req, timeout=90) as response:
                body = json.loads(response.read())
                status = response.status
        except urllib.error.HTTPError as exc:
            try:
                body = json.loads(exc.read())
            except Exception:
                body = {}
            status = exc.code
        except Exception as exc:
            body, status = {"error": str(exc)}, 0
        wall = time.perf_counter() - started
        usage = body.get("usage") or {}
        timing = body.get("strata") or {}
        count = int(usage.get("completion_tokens", 0) or 0)
        service_s = timing.get("seconds")
        return {
            "status": status,
            "wall_s": wall,
            "service_s": float(service_s) if isinstance(service_s, (int, float)) else None,
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": count,
            "end_to_end_tok_s": count / float(service_s) if count and service_s else None,
        }

    before = get_json("/metrics")
    warmup = completion(8)
    if warmup["status"] != 200:
        raise SystemExit(f"warmup failed with HTTP {warmup['status']}")
    print(f"Warmup complete: {warmup['completion_tokens']} tokens in {warmup['wall_s']:.2f}s")

    stop = threading.Event()
    hardware_samples: list[dict] = []

    def sample_metrics() -> None:
        while not stop.wait(1.0):
            try:
                hardware_samples.append(get_json("/metrics"))
            except Exception:
                continue

    sampler = threading.Thread(target=sample_metrics, daemon=True)
    sampler.start()
    levels = []
    for concurrency in CONCURRENCY_LEVELS:
        gate = threading.Barrier(concurrency)

        def worker() -> list[dict]:
            gate.wait(timeout=10)
            return [completion(MAX_TOKENS) for _ in range(REPEATS_PER_WORKER)]

        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(worker) for _ in range(concurrency)]
            rows = [row for future in futures for row in future.result()]
        duration = time.perf_counter() - started
        tokens = sum(r["completion_tokens"] for r in rows)
        latencies = [r["wall_s"] for r in rows]
        successes = sum(200 <= r["status"] < 300 for r in rows)
        level = {
            "concurrency": concurrency,
            "requests": len(rows),
            "successes": successes,
            "failures": len(rows) - successes,
            "duration_s": round(duration, 3),
            "requests_per_s": round(len(rows) / duration, 3) if duration else None,
            "completion_tokens": tokens,
            "completion_tokens_per_s": round(tokens / duration, 3) if duration else None,
            "client_latency_p50_s": round(percentile(latencies, 0.50), 3),
            "client_latency_p95_s": round(percentile(latencies, 0.95), 3),
            "client_latency_max_s": round(max(latencies), 3) if latencies else None,
            "mean_service_s": round(statistics.mean([r["service_s"] for r in rows if r["service_s"] is not None]), 3)
            if any(r["service_s"] is not None for r in rows) else None,
            "mean_end_to_end_tok_s": round(statistics.mean([r["end_to_end_tok_s"] for r in rows if r["end_to_end_tok_s"] is not None]), 3)
            if any(r["end_to_end_tok_s"] is not None for r in rows) else None,
        }
        levels.append(level)
        print(f"Concurrency {concurrency}: {successes}/{len(rows)} OK, {level['completion_tokens_per_s']} output tok/s aggregate, p95 {level['client_latency_p95_s']}s")

    stop.set()
    sampler.join(timeout=3)
    after = get_json("/metrics")
    max_active = max((x["requests"]["active"] for x in hardware_samples), default=0)
    device_ids = sorted({d["device"] for x in hardware_samples for d in x.get("devices", [])})
    peak_aicore = {}
    for device_id in device_ids:
        values = [d["aicore_percent"] for x in hardware_samples for d in x.get("devices", [])
                  if d["device"] == device_id and d.get("aicore_percent") is not None]
        peak_aicore[str(device_id)] = max(values) if values else None

    report = {
        "started_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "endpoint": BASE,
        "model": MODEL,
        "prompt_tokens_reported": warmup["prompt_tokens"],
        "max_tokens_per_request": MAX_TOKENS,
        "repeats_per_worker": REPEATS_PER_WORKER,
        "concurrency_levels": levels,
        "warmup": {k: v for k, v in warmup.items() if k != "status"} | {"status": warmup["status"]},
        "hardware_sampling": {
            "samples": len(hardware_samples),
            "peak_active_requests": max_active,
            "peak_aicore_percent_by_device": peak_aicore,
            "last_devices": after.get("devices", []),
        },
        "gateway_request_totals_before": before["requests"]["total"],
        "gateway_request_totals_after": after["requests"]["total"],
        "api_key_recorded": False,
        "metric_note": "Aggregate output tok/s uses generated tokens divided by wall time for each concurrency batch; request end-to-end rates include service overhead and queueing.",
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {OUT}")
    print(f"Gateway inference requests: {before['requests']['total']} -> {after['requests']['total']}; telemetry samples: {len(hardware_samples)}; peak active: {max_active}")


if __name__ == "__main__":
    main()
