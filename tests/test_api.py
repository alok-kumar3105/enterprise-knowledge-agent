import pytest
from fastapi.testclient import TestClient

from app.api import create_app


@pytest.fixture
def client(make_agent):
    agent = make_agent()
    with TestClient(create_app(agent=agent)) as c:
        yield c


@pytest.fixture
def secured(make_agent):
    agent = make_agent()
    agent.settings.api_key = "s3cret"
    with TestClient(create_app(agent=agent)) as c:
        yield c
    agent.settings.api_key = None


def test_health_and_ui(client):
    assert client.get("/health").json()["status"] == "ok"
    assert "Enterprise Knowledge Agent" in client.get("/").text


def test_ask_returns_answer_sources_and_metrics(client):
    r = client.post("/ask", json={"question": "How many days of annual leave do employees get?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] and body["route"] == "docs"
    assert body["sources"] and "latency_ms" in body["metrics"]
    assert client.get("/metrics").json()["requests"] == 1


def test_ask_validates_input(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "x" * 2001}).status_code == 422
    assert client.post("/ask", json={"question": "   "}).status_code == 400


def test_ingest_upload_and_rejects_bad_types(client):
    files = [
        ("files", ("../../evil name.md", b"# Bonus\n\nThe Helios bonus is paid in December.", "text/markdown")),
        ("files", ("payload.exe", b"MZ", "application/octet-stream")),
        ("files", ("kpis.csv", b"team,score\nA,1\nB,2\n", "text/csv")),
    ]
    r = client.post("/ingest", files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["documents"] == 1 and body["tables"] == {"kpis": 2}
    assert any("payload.exe" in s for s in body["skipped"])
    assert "kpis" in client.get("/stats").json()["tables"]


def test_api_key_is_enforced(secured):
    assert secured.get("/health").status_code == 200
    assert secured.post("/ask", json={"question": "hi"}).status_code == 401
    assert secured.get("/metrics", headers={"X-API-Key": "wrong"}).status_code == 401
    assert secured.get("/metrics", headers={"X-API-Key": "s3cret"}).status_code == 200
