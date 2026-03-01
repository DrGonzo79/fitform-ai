"""
Exercise telemetry API endpoints.

Handles real-time frame data from the edge device and provides
a Server-Sent Events stream for the live dashboard.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from models import FrameResponse, FrameTelemetry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/exercises", tags=["exercises"])

# In-memory latest frame store for SSE (per session)
_latest_frames: dict[str, dict[str, Any]] = {}


@router.post("/frame", response_model=FrameResponse)
async def submit_frame(frame: FrameTelemetry):
    """
    Receive a single frame of exercise telemetry from the edge device.

    Stores the frame data in the session store and updates the
    SSE stream for real-time dashboard updates.
    """
    from main import session_store  # Avoid circular import

    try:
        frame_number = session_store.add_frame(
            frame.session_id, frame.model_dump()
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    # Update SSE broadcast
    _latest_frames[frame.session_id] = {
        "exercise": frame.exercise.value,
        "rep_count": frame.rep_count,
        "phase": frame.phase.value,
        "confidence": frame.confidence,
        "angles": frame.angles,
        "rom_summary": frame.rom_summary,
        "timestamp": frame.timestamp,
    }

    return FrameResponse(
        session_id=frame.session_id,
        frame_number=frame_number,
    )


@router.get("/stream")
async def event_stream():
    """
    Server-Sent Events stream for real-time dashboard updates.

    Sends the latest frame telemetry every 200ms for all active sessions.
    Connect from the dashboard with:
        const source = new EventSource('/api/v1/exercises/stream');
    """

    async def generate():
        last_sent: dict[str, float] = {}
        while True:
            for session_id, data in _latest_frames.items():
                ts = data.get("timestamp", 0)
                if ts != last_sent.get(session_id, 0):
                    event_data = json.dumps({"session_id": session_id, **data})
                    yield f"data: {event_data}\n\n"
                    last_sent[session_id] = ts

            await asyncio.sleep(0.2)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
