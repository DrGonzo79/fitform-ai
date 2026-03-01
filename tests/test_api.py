"""
Integration tests for the FastAPI backend endpoints.

Tests session lifecycle, telemetry ingestion, and coaching
endpoint using httpx TestClient.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from main import app


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "azure_openai" in data


class TestSessionEndpoints:
    def test_create_session(self, client):
        resp = client.post("/api/v1/sessions", json={"started_at": 1700000000})
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert data["frame_count"] == 0

    def test_list_sessions(self, client):
        # Create a session first
        client.post("/api/v1/sessions", json={})
        resp = client.get("/api/v1/sessions")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_get_nonexistent_session(self, client):
        resp = client.get("/api/v1/sessions/nonexistent")
        assert resp.status_code == 404


class TestTelemetryEndpoints:
    def test_submit_frame(self, client):
        # Create session
        session = client.post("/api/v1/sessions", json={}).json()
        sid = session["session_id"]

        # Submit frame
        frame = {
            "session_id": sid,
            "timestamp": 1700000001,
            "exercise": "air_squat",
            "rep_count": 3,
            "phase": "up",
            "confidence": 0.85,
            "angles": {"left_knee": 90.0, "right_knee": 92.0},
            "rom_summary": {"left_knee": {"min": 85, "max": 170, "range": 85}},
        }
        resp = client.post("/api/v1/exercises/frame", json=frame)
        assert resp.status_code == 200
        assert resp.json()["frame_number"] == 1

    def test_submit_frame_invalid_session(self, client):
        frame = {
            "session_id": "nonexistent",
            "timestamp": 1700000001,
            "exercise": "air_squat",
        }
        resp = client.post("/api/v1/exercises/frame", json=frame)
        assert resp.status_code == 404


class TestCoachingEndpoint:
    def test_coaching_returns_feedback(self, client):
        # Create session with some data
        session = client.post("/api/v1/sessions", json={}).json()
        sid = session["session_id"]

        # Add a frame
        client.post("/api/v1/exercises/frame", json={
            "session_id": sid,
            "timestamp": 1700000001,
            "exercise": "air_squat",
            "rep_count": 5,
            "phase": "up",
            "confidence": 0.9,
            "angles": {"left_knee": 85},
            "rom_summary": {},
        })

        # Request coaching
        resp = client.post(f"/api/v1/sessions/{sid}/coach")
        assert resp.status_code == 200
        data = resp.json()
        assert "feedback" in data
        assert data["session_id"] == sid
