"""
Session analytics service.

Aggregates frame-level telemetry into session-level statistics
including exercise counts, ROM summaries, and trend analysis.
Uses in-memory storage for the MVP; production would use
Azure Cosmos DB or Azure SQL.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Optional

logger = logging.getLogger(__name__)


class SessionStore:
    """
    In-memory session and telemetry store.

    Stores exercise session data for the MVP. In production,
    this would be backed by Azure Cosmos DB with TTL policies.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}
        self._frames: dict[str, list[dict[str, Any]]] = {}

    def create_session(self, started_at: Optional[float] = None) -> str:
        """Create a new session and return its ID."""
        session_id = str(uuid.uuid4())[:8]
        self._sessions[session_id] = {
            "session_id": session_id,
            "started_at": started_at or time.time(),
            "ended_at": None,
            "frame_count": 0,
            "exercises": {},
            "rom_summary": {},
        }
        self._frames[session_id] = []
        logger.info("Session created: %s", session_id)
        return session_id

    def add_frame(self, session_id: str, frame_data: dict[str, Any]) -> int:
        """
        Add a telemetry frame to a session.

        Returns the frame number.
        """
        if session_id not in self._sessions:
            raise KeyError(f"Session {session_id} not found")

        self._frames[session_id].append(frame_data)
        session = self._sessions[session_id]
        session["frame_count"] += 1

        # Update exercise counts
        exercise = frame_data.get("exercise", "unknown")
        rep_count = frame_data.get("rep_count", 0)
        session["exercises"][exercise] = max(
            session["exercises"].get(exercise, 0), rep_count
        )

        # Update ROM summary (keep latest)
        if "rom_summary" in frame_data:
            session["rom_summary"] = frame_data["rom_summary"]

        return session["frame_count"]

    def get_session(self, session_id: str) -> Optional[dict[str, Any]]:
        """Get session metadata."""
        return self._sessions.get(session_id)

    def update_session(self, session_id: str, updates: dict[str, Any]) -> None:
        """Update session fields."""
        if session_id in self._sessions:
            self._sessions[session_id].update(updates)

    def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        """List recent sessions."""
        sessions = sorted(
            self._sessions.values(),
            key=lambda s: s["started_at"],
            reverse=True,
        )
        return sessions[:limit]

    def get_session_data(self, session_id: str) -> dict[str, Any]:
        """Get full session data including aggregated analytics."""
        session = self._sessions.get(session_id)
        if not session:
            raise KeyError(f"Session {session_id} not found")

        frames = self._frames.get(session_id, [])

        return {
            **session,
            "total_reps": sum(session["exercises"].values()),
            "frame_count": len(frames),
            "duration_seconds": (
                (session.get("ended_at") or time.time()) - session["started_at"]
            ),
        }

    def get_latest_frame(self, session_id: str) -> Optional[dict[str, Any]]:
        """Get the most recent frame for a session (for SSE streaming)."""
        frames = self._frames.get(session_id, [])
        return frames[-1] if frames else None
