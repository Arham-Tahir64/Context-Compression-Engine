def test_health_returns_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["version"] == "0.1.0"
    assert body["embedding_ready"] is True
    assert body["embedding_model"] == "fake-all-MiniLM-L6-v2"
    assert body["uptime_seconds"] >= 0


def test_compress_returns_structured_prompt(client):
    r = client.post("/compress", json={
        "project_hint": "/tmp/fake-project",
        "current_message": "What does this function do?",
        "recent_context": [],
        "metadata": {"tool": "test", "max_context_tokens": 4096},
    })
    assert r.status_code == 200
    body = r.json()
    assert "What does this function do?" in body["optimized_prompt"]
    assert "CONTEXT BRIEFING" in body["optimized_prompt"]
    assert body["warnings"] == []


def test_recall_returns_briefing(client):
    r = client.post("/recall", json={
        "project_hint": "/tmp/fake-project",
        "max_tokens": 1024,
    })
    assert r.status_code == 200
    assert "CONTEXT BRIEFING" in r.json()["briefing"]
