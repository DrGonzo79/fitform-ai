"""
Unit tests for the exercise classifier.

Tests rep counting logic, phase transitions, and exercise
type detection using synthetic landmark data.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add edge module to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "edge"))

from exercise_classifier import ExerciseClassifier, ExerciseType, Phase
from rom_calculator import JointAngles


class TestExerciseClassifier:
    """Test suite for ExerciseClassifier."""

    def setup_method(self):
        self.classifier = ExerciseClassifier(cooldown_frames=2)

    def _make_landmarks(self, nose_y=0.2, hip_y=0.5, ankle_y=0.9, shoulder_y=0.3):
        """Create synthetic landmarks with controlled positions."""
        lm = np.zeros((33, 3), dtype=np.float32)
        lm[:, 2] = 1.0  # All visible

        # Nose
        lm[0] = [0.5, nose_y, 1.0]
        # Shoulders
        lm[11] = [0.4, shoulder_y, 1.0]
        lm[12] = [0.6, shoulder_y, 1.0]
        # Hips
        lm[23] = [0.4, hip_y, 1.0]
        lm[24] = [0.6, hip_y, 1.0]
        # Knees
        lm[25] = [0.4, 0.7, 1.0]
        lm[26] = [0.6, 0.7, 1.0]
        # Ankles
        lm[27] = [0.4, ankle_y, 1.0]
        lm[28] = [0.6, ankle_y, 1.0]
        # Elbows
        lm[13] = [0.3, 0.35, 1.0]
        lm[14] = [0.7, 0.35, 1.0]
        # Wrists
        lm[15] = [0.2, 0.4, 1.0]
        lm[16] = [0.8, 0.4, 1.0]

        return lm

    def test_initial_state(self):
        """Classifier starts with zero reps for all exercises."""
        counts = self.classifier.get_all_counts()
        assert all(c == 0 for c in counts.values())

    def test_squat_rep_counting(self):
        """Squat reps counted on full down-up cycle."""
        landmarks = self._make_landmarks()

        # Simulate squat: standing → deep → standing
        standing_angles = JointAngles(left_knee=170, right_knee=170, left_hip=170, right_hip=170)
        deep_angles = JointAngles(left_knee=80, right_knee=80, left_hip=80, right_hip=80)

        # Start standing
        result = self.classifier.classify(landmarks, standing_angles)
        assert result.rep_count == 0

        # Go deep
        result = self.classifier.classify(landmarks, deep_angles)
        assert result.phase == Phase.DOWN

        # Come back up (need to tick cooldown)
        for _ in range(3):
            result = self.classifier.classify(landmarks, standing_angles)

        assert result.rep_count == 1

    def test_reset(self):
        """Reset clears all rep counts."""
        landmarks = self._make_landmarks()
        angles = JointAngles(left_knee=170, right_knee=170)

        self.classifier.classify(landmarks, angles)
        self.classifier.reset()

        counts = self.classifier.get_all_counts()
        assert all(c == 0 for c in counts.values())

    def test_classification_result_has_angles(self):
        """ClassificationResult includes angle data."""
        landmarks = self._make_landmarks()
        angles = JointAngles(left_knee=90, right_knee=90)

        result = self.classifier.classify(landmarks, angles)
        assert "left_knee" in result.angles
        assert result.angles["left_knee"] == 90.0


class TestPhaseTransitions:
    """Test phase transition logic."""

    def test_no_false_reps_without_full_cycle(self):
        """No rep counted if movement doesn't complete full cycle."""
        classifier = ExerciseClassifier(cooldown_frames=2)
        landmarks = np.zeros((33, 3), dtype=np.float32)
        landmarks[:, 2] = 1.0
        landmarks[0] = [0.5, 0.2, 1.0]  # Nose high
        landmarks[23] = [0.4, 0.5, 1.0]  # Hips
        landmarks[24] = [0.6, 0.5, 1.0]
        landmarks[27] = [0.4, 0.9, 1.0]  # Ankles
        landmarks[28] = [0.6, 0.9, 1.0]

        # Only go partially down (not past threshold)
        partial_angles = JointAngles(left_knee=120, right_knee=120)

        for _ in range(20):
            result = classifier.classify(landmarks, partial_angles)

        assert result.rep_count == 0
