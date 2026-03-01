"""
Pose estimation module using MediaPipe BlazePose.

Extracts 33 body landmarks from RGB frames captured by the OAK-D Lite.
Returns normalized and pixel-space coordinates for downstream processing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from config import EdgeConfig

logger = logging.getLogger(__name__)


# MediaPipe landmark indices for key joints
class LandmarkIndex:
    """Named constants for MediaPipe BlazePose landmark indices."""

    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28


@dataclass
class PoseResult:
    """Container for pose estimation output."""

    landmarks: Optional[list] = None  # MediaPipe NormalizedLandmarkList
    landmark_array: Optional[np.ndarray] = None  # (33, 3) array [x, y, visibility]
    detected: bool = False
    confidence: float = 0.0


class PoseEstimator:
    """
    Real-time pose estimation using MediaPipe BlazePose.

    Optimized for Jetson Orin Nano GPU acceleration. Processes
    individual RGB frames and returns 33-point body landmarks.

    Args:
        config: Edge configuration instance.
    """

    def __init__(self, config: EdgeConfig) -> None:
        self.config = config
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=config.model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=config.confidence_threshold,
            min_tracking_confidence=config.confidence_threshold,
        )

        logger.info(
            "PoseEstimator initialized | complexity=%d threshold=%.2f",
            config.model_complexity,
            config.confidence_threshold,
        )

    def process_frame(self, frame: np.ndarray) -> PoseResult:
        """
        Run pose estimation on a single BGR frame.

        Args:
            frame: BGR image from camera (H, W, 3).

        Returns:
            PoseResult with landmarks if a pose is detected.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            return PoseResult(detected=False)

        # Convert to numpy array for fast math
        landmarks = results.pose_landmarks.landmark
        arr = np.array(
            [[lm.x, lm.y, lm.visibility] for lm in landmarks],
            dtype=np.float32,
        )

        avg_visibility = float(np.mean(arr[:, 2]))

        return PoseResult(
            landmarks=results.pose_landmarks,
            landmark_array=arr,
            detected=True,
            confidence=avg_visibility,
        )

    def draw_landmarks(self, frame: np.ndarray, result: PoseResult) -> np.ndarray:
        """
        Draw pose skeleton overlay on the frame.

        Args:
            frame: BGR image to annotate.
            result: PoseResult from process_frame().

        Returns:
            Annotated frame copy.
        """
        annotated = frame.copy()
        if result.detected and result.landmarks:
            self.mp_draw.draw_landmarks(
                annotated,
                result.landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_styles.get_default_pose_landmarks_style(),
            )
        return annotated

    def close(self) -> None:
        """Release MediaPipe resources."""
        self.pose.close()
        logger.info("PoseEstimator closed.")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
