"""End-to-end integration tests for the /compress and /recall endpoints."""
from __future__ import annotations

from unittest.mock import patch


def test_compress_returns_real_project_id(client):
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    r = client.post("/compress", json={
        "project_hint": project_root,
        "current_message": "What does the chunker do?",
        "recent_context": [],
        "metadata": {"tool": "test", "max_context_tokens": 4096},
    })
    assert r.status_code == 200
    body = r.json()
    assert body["project_id"] != "__stub__"
    assert len(body["project_id"]) == 16
    assert body["optimized_prompt"]
    assert body["warnings"] == []


def test_compress_with_context_routes_to_memory(client):
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    r = client.post("/compress", json={
        "project_hint": project_root,
        "current_message": "Fix the calculate function",
        "recent_context": [
            {"role": "user", "content": "def calculate(x): return x * 2", "turn_index": 0},
            {"role": "assistant", "content": "The calculate function doubles its input.", "turn_index": 1},
        ],
        "metadata": {"tool": "test", "max_context_tokens": 4096},
    })
    assert r.status_code == 200
    body = r.json()
    assert body["original_token_estimate"] > 0
    assert body["latency_ms"]["total_ms"] > 0


def test_recall_returns_briefing(client):
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    r = client.post("/recall", json={
        "project_hint": project_root,
        "max_tokens": 1024,
    })
    assert r.status_code == 200
    body = r.json()
    assert "CONTEXT BRIEFING" in body["briefing"]
    assert body["project_id"] != "__stub__"
    assert body["latency_ms"] >= 0


def test_project_stats_returns_counts(client):
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    from cce.identity.resolver import resolve_project_id
    project_id = resolve_project_id(project_root)
    r = client.get(f"/project/{project_id}/stats")
    assert r.status_code == 200
    body = r.json()
    assert "stm_records" in body
    assert "wm_records" in body


def test_compress_latency_under_threshold(client):
    """Embedding call dominates; ensure overhead is logged and reasonable."""
    import os, time
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    t0 = time.time()
    r = client.post("/compress", json={
        "project_hint": project_root,
        "current_message": "quick latency test",
        "recent_context": [],
        "metadata": {"tool": "test", "max_context_tokens": 4096},
    })
    elapsed = (time.time() - t0) * 1000
    assert r.status_code == 200
    # Wall clock should be well under 5s even on first cold embed
    assert elapsed < 5000
