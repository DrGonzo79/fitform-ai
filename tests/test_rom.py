"""
Unit tests for the Range of Motion calculator.

Tests angle calculation accuracy, smoothing behavior,
and ROM statistics tracking.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "edge"))

from rom_calculator import ROMCalculator


class TestAngleCalculation:
    """Test core angle calculation geometry."""

    def test_right_angle(self):
        """90-degree angle calculated correctly."""
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 0.0])
        c = np.array([0.0, 1.0])

        angle = ROMCalculator._calculate_angle(a, b, c)
        assert abs(angle - 90.0) < 0.1

    def test_straight_angle(self):
        """180-degree (straight) angle calculated correctly."""
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        c = np.array([2.0, 0.0])

        angle = ROMCalculator._calculate_angle(a, b, c)
        assert abs(angle - 180.0) < 0.1

    def test_acute_angle(self):
        """45-degree angle calculated correctly."""
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 0.0])
        c = np.array([1.0, 1.0])

        angle = ROMCalculator._calculate_angle(a, b, c)
        assert abs(angle - 45.0) < 0.5

    def test_zero_length_vector_handled(self):
        """Coincident points don't cause division by zero."""
        a = np.array([0.0, 0.0])
        b = np.array([0.0, 0.0])
        c = np.array([1.0, 0.0])

        # Should not raise
        angle = ROMCalculator._calculate_angle(a, b, c)
        assert 0 <= angle <= 180


class TestROMStatistics:
    """Test ROM tracking across frames."""

    def setup_method(self):
        self.calc = ROMCalculator(smoothing_window=1)

    def _make_landmarks(self, knee_angle_approx: float):
        """Create landmarks that produce approximately the target knee angle."""
        lm = np.zeros((33, 3), dtype=np.float32)
        lm[:, 2] = 1.0

        # Position hip-knee-ankle to achieve target angle
        rad = np.radians(knee_angle_approx)
        lm[23] = [0.5, 0.3, 1.0]  # Left hip
        lm[24] = [0.5, 0.3, 1.0]  # Right hip
        lm[25] = [0.5, 0.5, 1.0]  # Left knee
        lm[26] = [0.5, 0.5, 1.0]  # Right knee
        # Ankle position based on target angle
        lm[27] = [0.5 + 0.2 * np.sin(rad), 0.5 + 0.2 * np.cos(rad), 1.0]
        lm[28] = [0.5 + 0.2 * np.sin(rad), 0.5 + 0.2 * np.cos(rad), 1.0]

        # Shoulders
        lm[11] = [0.4, 0.2, 1.0]
        lm[12] = [0.6, 0.2, 1.0]
        # Elbows
        lm[13] = [0.3, 0.3, 1.0]
        lm[14] = [0.7, 0.3, 1.0]
        # Wrists
        lm[15] = [0.2, 0.4, 1.0]
        lm[16] = [0.8, 0.4, 1.0]

        return lm

    def test_rom_tracks_min_max(self):
        """ROM stats capture min and max angles over time."""
        for angle in [150, 120, 90, 120, 150]:
            lm = self._make_landmarks(angle)
            self.calc.compute_angles(lm)

        summary = self.calc.get_rom_summary()
        # At least left_knee should show some range
        assert summary["left_knee"]["range"] > 0

    def test_reset_clears_stats(self):
        """Reset returns all ROM stats to defaults."""
        lm = self._make_landmarks(90)
        self.calc.compute_angles(lm)
        self.calc.reset()

        summary = self.calc.get_rom_summary()
        assert summary["left_knee"]["min"] == 180.0
        assert summary["left_knee"]["max"] == 0.0


class TestSmoothing:
    """Test angle smoothing behavior."""

    def test_smoothing_reduces_noise(self):
        """Smoothed output has less variance than noisy input."""
        calc = ROMCalculator(smoothing_window=5)
        lm = np.zeros((33, 3), dtype=np.float32)
        lm[:, 2] = 1.0

        # Set up a basic arm configuration
        lm[11] = [0.5, 0.2, 1.0]
        lm[13] = [0.5, 0.4, 1.0]
        lm[15] = [0.5, 0.6, 1.0]
        lm[12] = [0.5, 0.2, 1.0]
        lm[14] = [0.5, 0.4, 1.0]
        lm[16] = [0.5, 0.6, 1.0]
        lm[23] = [0.5, 0.5, 1.0]
        lm[24] = [0.5, 0.5, 1.0]
        lm[25] = [0.5, 0.7, 1.0]
        lm[26] = [0.5, 0.7, 1.0]
        lm[27] = [0.5, 0.9, 1.0]
        lm[28] = [0.5, 0.9, 1.0]

        # Feed 10 frames and check output doesn't spike
        results = []
        for _ in range(10):
            # Add small noise
            noisy = lm.copy()
            noisy[:, :2] += np.random.normal(0, 0.005, (33, 2)).astype(np.float32)
            angles = calc.compute_angles(noisy)
            results.append(angles.left_elbow)

        # After 5 frames, output should be relatively stable
        variance = np.var(results[5:])
        assert variance < 50  # Reasonable stability
