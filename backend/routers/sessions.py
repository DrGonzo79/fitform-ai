"""
Session management API endpoints.

Handles session lifecycle (create, read, update) and AI coaching
feedback generation via Azure OpenAI.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from models import (
    CoachingResponse,
    SessionCreate,
    SessionResponse,
    SessionUpdate,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/sessions", tags=["sessions"])


@router.post("", response_model=SessionResponse)
async def create_session(body: SessionCreate):
    """Create a new exercise session."""
    from main import session_store

    session_id = session_store.create_session(started_at=body.started_at)
    session = session_store.get_session(session_id)
    return SessionResponse(**session)


@router.get("", response_model=list[SessionResponse])
async def list_sessions():
    """List all exercise sessions (most recent first)."""
    from main import session_store

    sessions = session_store.list_sessions()
    return [SessionResponse(**s) for s in sessions]


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """Get session detail including exercise counts and ROM analytics."""
    from main import session_store

    session = session_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionResponse(**session)


@router.patch("/{session_id}")
async def update_session(session_id: str, body: SessionUpdate):
    """Update session (e.g., mark as ended)."""
    from main import session_store

    session = session_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session_store.update_session(session_id, body.model_dump(exclude_unset=True))
    return {"status": "updated"}


@router.post("/{session_id}/coach", response_model=CoachingResponse)
async def get_coaching(session_id: str):
    """
    Generate AI coaching feedback for a session.

    Uses Azure OpenAI (GPT-4o) to analyze exercise data and
    produce personalized form corrections and recommendations.
    """
    from main import ai_coach, session_store

    try:
        session_data = session_store.get_session_data(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    result = ai_coach.generate_feedback(session_data)

    return CoachingResponse(
        session_id=session_id,
        feedback=result.get("feedback", ""),
        form_score=result.get("form_score"),
        recommendations=result.get("recommendations", []),
        model=result.get("model", ""),
    )
