def test_health_returns_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["version"] == "0.1.0"
    assert body["uptime_seconds"] >= 0


def test_compress_passthrough(client):
    r = client.post("/compress", json={
        "project_hint": "/tmp/fake-project",
        "current_message": "What does this function do?",
        "recent_context": [],
        "metadata": {"tool": "test", "max_context_tokens": 4096},
    })
    assert r.status_code == 200
    body = r.json()
    assert body["optimized_prompt"] == "What does this function do?"
    assert body["compression_ratio"] == 1.0
    assert "pipeline not yet wired" in body["warnings"][0]


def test_recall_stub(client):
    r = client.post("/recall", json={
        "project_hint": "/tmp/fake-project",
        "max_tokens": 1024,
    })
    assert r.status_code == 200
    assert "not yet loaded" in r.json()["briefing"]
