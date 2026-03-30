#!/usr/bin/env python
"""Latency benchmark for the Context Compression Engine.

Measures end-to-end, retrieval, and assembly latency across multiple
request sizes. Run with the server already started:

    source .venv/bin/activate
    python scripts/bench_latency.py [--host 127.0.0.1] [--port 8765] [--runs 20]
"""
from __future__ import annotations

import argparse
import statistics
import time

import httpx

_SAMPLE_CONTEXT = [
    {"role": "user", "content": "Let's implement a caching layer for the API.", "turn_index": 0},
    {"role": "assistant", "content": "Good idea. We can use Redis with a TTL of 60 seconds for most endpoints.", "turn_index": 1},
    {"role": "user", "content": "What about cache invalidation when the DB changes?", "turn_index": 2},
    {"role": "assistant", "content": "We should publish events on write and have the cache layer subscribe to invalidate affected keys.", "turn_index": 3},
    {"role": "user", "content": "Can you write the CacheManager class?", "turn_index": 4},
    {"role": "assistant", "content": "```python\nclass CacheManager:\n    def __init__(self, redis_url: str, ttl: int = 60):\n        self._client = redis.from_url(redis_url)\n        self._ttl = ttl\n\n    def get(self, key: str):\n        return self._client.get(key)\n\n    def set(self, key: str, value: str):\n        self._client.setex(key, self._ttl, value)\n\n    def invalidate(self, key: str):\n        self._client.delete(key)\n```", "turn_index": 5},
]


def bench_compress(base_url: str, n_context_turns: int, runs: int) -> dict:
    context = _SAMPLE_CONTEXT[:n_context_turns]
    latencies = []
    retrieval_latencies = []
    assembly_latencies = []
    compression_ratios = []

    for _ in range(runs):
        r = httpx.post(
            f"{base_url}/compress",
            json={
                "project_hint": "bench-session",
                "current_message": "How does the cache invalidation work?",
                "recent_context": context,
                "metadata": {"tool": "bench", "max_context_tokens": 4096},
            },
            timeout=30.0,
        )
        r.raise_for_status()
        body = r.json()
        latencies.append(body["latency_ms"]["total_ms"])
        retrieval_latencies.append(body["latency_ms"]["retrieval_ms"])
        assembly_latencies.append(body["latency_ms"]["assembly_ms"])
        compression_ratios.append(body["compression_ratio"])

    return {
        "context_turns": n_context_turns,
        "runs": runs,
        "total_ms": {
            "mean": round(statistics.mean(latencies), 2),
            "p50": round(statistics.median(latencies), 2),
            "p95": round(sorted(latencies)[int(0.95 * len(latencies))], 2),
            "max": round(max(latencies), 2),
        },
        "retrieval_ms_mean": round(statistics.mean(retrieval_latencies), 2),
        "assembly_ms_mean": round(statistics.mean(assembly_latencies), 2),
        "compression_ratio_mean": round(statistics.mean(compression_ratios), 2),
    }


def bench_recall(base_url: str, runs: int) -> dict:
    latencies = []
    for _ in range(runs):
        t0 = time.time()
        r = httpx.post(
            f"{base_url}/recall",
            json={"project_hint": "bench-session", "max_tokens": 2048},
            timeout=30.0,
        )
        r.raise_for_status()
        latencies.append(r.json()["latency_ms"])

    return {
        "runs": runs,
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 2),
            "p50": round(statistics.median(latencies), 2),
            "p95": round(sorted(latencies)[int(0.95 * len(latencies))], 2),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="CCE latency benchmark")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args()
    base_url = f"http://{args.host}:{args.port}"

    print(f"\n{'='*60}")
    print(f"  Context Compression Engine — Latency Benchmark")
    print(f"  Target: {base_url}   Runs per scenario: {args.runs}")
    print(f"{'='*60}\n")

    # Warm up (first call loads the embedding model)
    print("Warming up (loading embedding model)...")
    httpx.post(f"{base_url}/compress", json={
        "project_hint": "bench-session",
        "current_message": "warmup",
        "recent_context": [],
        "metadata": {"tool": "bench", "max_context_tokens": 4096},
    }, timeout=60.0)
    print("Ready.\n")

    print(f"{'Scenario':<30} {'Mean':>8} {'P50':>8} {'P95':>8} {'Max':>8}  {'Ratio':>6}")
    print("-" * 75)

    for n_turns in [0, 2, 4, 6]:
        label = f"/compress ({n_turns} context turns)"
        result = bench_compress(base_url, n_turns, args.runs)
        t = result["total_ms"]
        print(
            f"{label:<30} {t['mean']:>7.1f}ms {t['p50']:>7.1f}ms "
            f"{t['p95']:>7.1f}ms {t['max']:>7.1f}ms  {result['compression_ratio_mean']:>5.1f}x"
        )

    recall_result = bench_recall(base_url, args.runs)
    t = recall_result["latency_ms"]
    print(
        f"{'  /recall':<30} {t['mean']:>7.1f}ms {t['p50']:>7.1f}ms "
        f"{t['p95']:>7.1f}ms {'—':>8}"
    )

    print(f"\n{'='*60}")
    print("Target: <500ms end-to-end overhead per request")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
