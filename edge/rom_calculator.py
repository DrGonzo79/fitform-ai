"""
Range of Motion (ROM) calculator.

Computes joint angles from MediaPipe pose landmarks using vector
geometry. Provides real-time angle tracking and per-rep min/max
range analysis for biomechanical assessment.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from pose_estimator import LandmarkIndex

logger = logging.getLogger(__name__)


@dataclass
class JointAngles:
    """Current joint angle measurements in degrees."""

    left_knee: float = 0.0
    right_knee: float = 0.0
    left_hip: float = 0.0
    right_hip: float = 0.0
    left_elbow: float = 0.0
    right_elbow: float = 0.0
    left_shoulder: float = 0.0
    right_shoulder: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "left_knee": round(self.left_knee, 1),
            "right_knee": round(self.right_knee, 1),
            "left_hip": round(self.left_hip, 1),
            "right_hip": round(self.right_hip, 1),
            "left_elbow": round(self.left_elbow, 1),
            "right_elbow": round(self.right_elbow, 1),
            "left_shoulder": round(self.left_shoulder, 1),
            "right_shoulder": round(self.right_shoulder, 1),
        }


@dataclass
class ROMStats:
    """Min/max range of motion statistics for a joint across a session."""

    min_angle: float = 180.0
    max_angle: float = 0.0

    def update(self, angle: float) -> None:
        self.min_angle = min(self.min_angle, angle)
        self.max_angle = max(self.max_angle, angle)

    @property
    def range(self) -> float:
        return self.max_angle - self.min_angle


class ROMCalculator:
    """
    Computes joint angles and tracks range-of-motion statistics.

    Uses 3-point angle calculation (law of cosines / atan2) on
    MediaPipe landmark coordinates to derive anatomical joint angles.

    Args:
        smoothing_window: Number of frames for moving average smoothing.
    """

    # Joint definitions: (proximal, joint_center, distal) landmark indices
    JOINT_DEFINITIONS = {
        "left_knee": (LandmarkIndex.LEFT_HIP, LandmarkIndex.LEFT_KNEE, LandmarkIndex.LEFT_ANKLE),
        "right_knee": (LandmarkIndex.RIGHT_HIP, LandmarkIndex.RIGHT_KNEE, LandmarkIndex.RIGHT_ANKLE),
        "left_hip": (LandmarkIndex.LEFT_SHOULDER, LandmarkIndex.LEFT_HIP, LandmarkIndex.LEFT_KNEE),
        "right_hip": (LandmarkIndex.RIGHT_SHOULDER, LandmarkIndex.RIGHT_HIP, LandmarkIndex.RIGHT_KNEE),
        "left_elbow": (LandmarkIndex.LEFT_SHOULDER, LandmarkIndex.LEFT_ELBOW, LandmarkIndex.LEFT_WRIST),
        "right_elbow": (LandmarkIndex.RIGHT_SHOULDER, LandmarkIndex.RIGHT_ELBOW, LandmarkIndex.RIGHT_WRIST),
        "left_shoulder": (LandmarkIndex.LEFT_HIP, LandmarkIndex.LEFT_SHOULDER, LandmarkIndex.LEFT_ELBOW),
        "right_shoulder": (LandmarkIndex.RIGHT_HIP, LandmarkIndex.RIGHT_SHOULDER, LandmarkIndex.RIGHT_ELBOW),
    }

    def __init__(self, smoothing_window: int = 5) -> None:
        self._buffers: dict[str, deque] = {
            name: deque(maxlen=smoothing_window)
            for name in self.JOINT_DEFINITIONS
        }
        self.rom_stats: dict[str, ROMStats] = {
            name: ROMStats() for name in self.JOINT_DEFINITIONS
        }

    @staticmethod
    def _calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """
        Calculate the angle at point B formed by points A-B-C.

        Uses atan2 for numerical stability.

        Args:
            a: Proximal point (x, y).
            b: Joint center point (x, y).
            c: Distal point (x, y).

        Returns:
            Angle in degrees [0, 180].
        """
        ba = a - b
        bc = c - b

        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        cosine = np.clip(cosine, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine))

        return float(angle)

    def compute_angles(self, landmarks: np.ndarray) -> JointAngles:
        """
        Compute all joint angles from a landmark array.

        Args:
            landmarks: (33, 3) array from PoseResult.landmark_array.

        Returns:
            JointAngles with smoothed angle values.
        """
        angles = {}

        for name, (idx_a, idx_b, idx_c) in self.JOINT_DEFINITIONS.items():
            a = landmarks[idx_a, :2]
            b = landmarks[idx_b, :2]
            c = landmarks[idx_c, :2]

            raw_angle = self._calculate_angle(a, b, c)

            # Apply moving average smoothing
            self._buffers[name].append(raw_angle)
            smoothed = float(np.mean(self._buffers[name]))

            # Update ROM stats
            self.rom_stats[name].update(smoothed)

            angles[name] = smoothed

        return JointAngles(**angles)

    def get_rom_summary(self) -> dict[str, dict[str, float]]:
        """
        Get ROM statistics summary for all tracked joints.

        Returns:
            Dict mapping joint name to {min, max, range} in degrees.
        """
        return {
            name: {
                "min": round(stats.min_angle, 1),
                "max": round(stats.max_angle, 1),
                "range": round(stats.range, 1),
            }
            for name, stats in self.rom_stats.items()
        }

    def reset(self) -> None:
        """Reset all ROM statistics and smoothing buffers."""
        for buf in self._buffers.values():
            buf.clear()
        self.rom_stats = {
            name: ROMStats() for name in self.JOINT_DEFINITIONS
        }
