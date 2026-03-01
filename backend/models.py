"""
Pydantic data models for the FitForm AI API.

Defines request/response schemas for exercise telemetry,
session management, and coaching feedback endpoints.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ExerciseType(str, Enum):
    AIR_SQUAT = "air_squat"
    PUSH_UP = "push_up"
    SIT_UP = "sit_up"
    UNKNOWN = "unknown"


class Phase(str, Enum):
    UP = "up"
    DOWN = "down"
    NEUTRAL = "neutral"


# ---------------------------------------------------------------------------
# Exercise Telemetry
# ---------------------------------------------------------------------------

class FrameTelemetry(BaseModel):
    """Single frame of exercise data from the edge device."""

    session_id: str
    timestamp: float
    exercise: ExerciseType
    rep_count: int = 0
    phase: Phase = Phase.NEUTRAL
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    angles: dict[str, float] = Field(default_factory=dict)
    rom_summary: dict[str, dict[str, float]] = Field(default_factory=dict)


class FrameResponse(BaseModel):
    """Acknowledgement of a received frame."""

    status: str = "ok"
    session_id: str
    frame_number: int


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

class SessionCreate(BaseModel):
    """Request to create a new exercise session."""

    started_at: Optional[float] = None


class SessionResponse(BaseModel):
    """Session metadata."""

    session_id: str
    started_at: float
    ended_at: Optional[float] = None
    frame_count: int = 0
    exercises: dict[str, int] = Field(default_factory=dict)
    rom_summary: dict[str, Any] = Field(default_factory=dict)


class SessionUpdate(BaseModel):
    """Partial session update."""

    ended_at: Optional[float] = None


# ---------------------------------------------------------------------------
# Coaching
# ---------------------------------------------------------------------------

class CoachingRequest(BaseModel):
    """Request for AI-generated coaching feedback."""

    focus_areas: Optional[list[str]] = None


class CoachingResponse(BaseModel):
    """AI coaching feedback for a session."""

    session_id: str
    feedback: str
    form_score: Optional[float] = Field(None, ge=0.0, le=10.0)
    recommendations: list[str] = Field(default_factory=list)
    model: str = ""
