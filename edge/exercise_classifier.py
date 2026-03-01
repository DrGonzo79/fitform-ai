"""
Exercise classifier with repetition counting.

Uses joint-angle heuristics derived from biomechanical research
to classify exercises and count reps in real time. Each exercise
is defined by characteristic angle patterns and body orientation.

Supported exercises:
- Air Squat: Knee flexion/extension cycle with upright torso
- Push-Up: Elbow flexion/extension in prone position
- Sit-Up: Hip flexion/extension from supine position
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from pose_estimator import LandmarkIndex
from rom_calculator import JointAngles

logger = logging.getLogger(__name__)


class ExerciseType(str, Enum):
    """Supported exercise types."""

    AIR_SQUAT = "air_squat"
    PUSH_UP = "push_up"
    SIT_UP = "sit_up"
    UNKNOWN = "unknown"


class Phase(str, Enum):
    """Movement phase within a repetition."""

    UP = "up"
    DOWN = "down"
    NEUTRAL = "neutral"


@dataclass
class RepState:
    """Tracks repetition state for a single exercise type."""

    count: int = 0
    phase: Phase = Phase.NEUTRAL
    cooldown: int = 0  # Frames remaining before next rep can count
    peak_angle: float = 180.0  # Track depth of each rep
    rep_angles: list[float] = field(default_factory=list)


@dataclass
class ClassificationResult:
    """Output of a single frame's classification."""

    exercise: ExerciseType
    confidence: float
    rep_count: int
    phase: Phase
    angles: dict[str, float]


class ExerciseClassifier:
    """
    Rule-based exercise classifier with repetition counting.

    Classifies exercises based on body orientation and joint angle
    patterns. Uses hysteresis thresholds to prevent false rep counts.

    Args:
        cooldown_frames: Minimum frames between rep increments.
    """

    # --- Angle thresholds (degrees) ---
    # Squat: knee angle < threshold = bottom position
    SQUAT_DOWN_THRESH = 100.0
    SQUAT_UP_THRESH = 155.0

    # Push-up: elbow angle < threshold = bottom position
    PUSHUP_DOWN_THRESH = 100.0
    PUSHUP_UP_THRESH = 155.0

    # Sit-up: hip angle < threshold = up position
    SITUP_UP_THRESH = 70.0
    SITUP_DOWN_THRESH = 130.0

    def __init__(self, cooldown_frames: int = 10) -> None:
        self.cooldown_frames = cooldown_frames
        self._states: dict[ExerciseType, RepState] = {
            ExerciseType.AIR_SQUAT: RepState(),
            ExerciseType.PUSH_UP: RepState(),
            ExerciseType.SIT_UP: RepState(),
        }
        self._current_exercise: ExerciseType = ExerciseType.UNKNOWN

    def classify(
        self,
        landmarks: np.ndarray,
        angles: JointAngles,
    ) -> ClassificationResult:
        """
        Classify the current exercise and update rep count.

        Args:
            landmarks: (33, 3) MediaPipe landmark array.
            angles: Computed joint angles from ROMCalculator.

        Returns:
            ClassificationResult with exercise type, rep count, and phase.
        """
        # Step 1: Determine body orientation to narrow exercise type
        exercise = self._detect_exercise_type(landmarks, angles)
        self._current_exercise = exercise

        # Step 2: Update rep counting for detected exercise
        phase = Phase.NEUTRAL
        rep_count = 0
        confidence = 0.0

        if exercise == ExerciseType.AIR_SQUAT:
            phase, rep_count, confidence = self._update_squat(angles)
        elif exercise == ExerciseType.PUSH_UP:
            phase, rep_count, confidence = self._update_pushup(angles)
        elif exercise == ExerciseType.SIT_UP:
            phase, rep_count, confidence = self._update_situp(angles)

        # Tick cooldowns
        for state in self._states.values():
            if state.cooldown > 0:
                state.cooldown -= 1

        return ClassificationResult(
            exercise=exercise,
            confidence=confidence,
            rep_count=rep_count,
            phase=phase,
            angles=angles.to_dict(),
        )

    def _detect_exercise_type(
        self,
        landmarks: np.ndarray,
        angles: JointAngles,
    ) -> ExerciseType:
        """
        Determine exercise type from body orientation and posture.

        Uses relative positions of nose, hips, and shoulders to infer
        whether the person is upright, prone, or supine.
        """
        nose_y = landmarks[LandmarkIndex.NOSE, 1]
        hip_y = np.mean([
            landmarks[LandmarkIndex.LEFT_HIP, 1],
            landmarks[LandmarkIndex.RIGHT_HIP, 1],
        ])
        shoulder_y = np.mean([
            landmarks[LandmarkIndex.LEFT_SHOULDER, 1],
            landmarks[LandmarkIndex.RIGHT_SHOULDER, 1],
        ])
        ankle_y = np.mean([
            landmarks[LandmarkIndex.LEFT_ANKLE, 1],
            landmarks[LandmarkIndex.RIGHT_ANKLE, 1],
        ])

        # Vertical spread: how "tall" the person appears in frame
        # (In normalized coords, y increases downward)
        vertical_spread = abs(nose_y - ankle_y)

        # If nose is significantly above hips → upright → likely squat
        if nose_y < hip_y - 0.1 and vertical_spread > 0.3:
            return ExerciseType.AIR_SQUAT

        # If body is mostly horizontal (small vertical spread)
        if vertical_spread < 0.25:
            # Distinguish prone (push-up) vs supine (sit-up)
            # In push-up: wrists are roughly below shoulders
            # In sit-up: hips are primary flexion point
            avg_elbow = (angles.left_elbow + angles.right_elbow) / 2
            avg_hip = (angles.left_hip + angles.right_hip) / 2

            if avg_elbow < 160 and avg_hip > 120:
                return ExerciseType.PUSH_UP
            elif avg_hip < 140:
                return ExerciseType.SIT_UP

        # Fallback: use the most active exercise based on angle movement
        return self._current_exercise if self._current_exercise != ExerciseType.UNKNOWN else ExerciseType.AIR_SQUAT

    def _update_squat(self, angles: JointAngles) -> tuple[Phase, int, float]:
        """Update squat rep state based on knee angles."""
        state = self._states[ExerciseType.AIR_SQUAT]
        avg_knee = (angles.left_knee + angles.right_knee) / 2

        phase = state.phase

        if avg_knee < self.SQUAT_DOWN_THRESH:
            phase = Phase.DOWN
            state.peak_angle = min(state.peak_angle, avg_knee)
        elif avg_knee > self.SQUAT_UP_THRESH and state.phase == Phase.DOWN:
            if state.cooldown <= 0:
                state.count += 1
                state.rep_angles.append(state.peak_angle)
                state.cooldown = self.cooldown_frames
                state.peak_angle = 180.0
                logger.info("Squat rep #%d (depth: %.1f°)", state.count, state.rep_angles[-1])
            phase = Phase.UP

        state.phase = phase
        confidence = min(1.0, max(0.3, 1.0 - abs(avg_knee - 90) / 90))
        return phase, state.count, confidence

    def _update_pushup(self, angles: JointAngles) -> tuple[Phase, int, float]:
        """Update push-up rep state based on elbow angles."""
        state = self._states[ExerciseType.PUSH_UP]
        avg_elbow = (angles.left_elbow + angles.right_elbow) / 2

        phase = state.phase

        if avg_elbow < self.PUSHUP_DOWN_THRESH:
            phase = Phase.DOWN
            state.peak_angle = min(state.peak_angle, avg_elbow)
        elif avg_elbow > self.PUSHUP_UP_THRESH and state.phase == Phase.DOWN:
            if state.cooldown <= 0:
                state.count += 1
                state.rep_angles.append(state.peak_angle)
                state.cooldown = self.cooldown_frames
                state.peak_angle = 180.0
                logger.info("Push-up rep #%d (depth: %.1f°)", state.count, state.rep_angles[-1])
            phase = Phase.UP

        state.phase = phase
        confidence = min(1.0, max(0.3, 1.0 - abs(avg_elbow - 90) / 90))
        return phase, state.count, confidence

    def _update_situp(self, angles: JointAngles) -> tuple[Phase, int, float]:
        """Update sit-up rep state based on hip angles."""
        state = self._states[ExerciseType.SIT_UP]
        avg_hip = (angles.left_hip + angles.right_hip) / 2

        phase = state.phase

        if avg_hip < self.SITUP_UP_THRESH:
            phase = Phase.UP
            state.peak_angle = min(state.peak_angle, avg_hip)
        elif avg_hip > self.SITUP_DOWN_THRESH and state.phase == Phase.UP:
            if state.cooldown <= 0:
                state.count += 1
                state.rep_angles.append(state.peak_angle)
                state.cooldown = self.cooldown_frames
                state.peak_angle = 180.0
                logger.info("Sit-up rep #%d (depth: %.1f°)", state.count, state.rep_angles[-1])
            phase = Phase.DOWN

        state.phase = phase
        confidence = min(1.0, max(0.3, 1.0 - abs(avg_hip - 60) / 60))
        return phase, state.count, confidence

    def get_all_counts(self) -> dict[str, int]:
        """Get rep counts for all exercise types."""
        return {
            ex.value: state.count
            for ex, state in self._states.items()
        }

    def reset(self) -> None:
        """Reset all rep states."""
        for state in self._states.values():
            state.count = 0
            state.phase = Phase.NEUTRAL
            state.cooldown = 0
            state.peak_angle = 180.0
            state.rep_angles.clear()
