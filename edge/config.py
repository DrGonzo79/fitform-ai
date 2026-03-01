"""
Edge device configuration management.

Loads settings from environment variables with sensible defaults
for the Jetson Orin Nano + OAK-D Lite hardware configuration.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)


@dataclass
class EdgeConfig:
    """Configuration for the edge processing pipeline."""

    # --- Backend connection ---
    backend_url: str = field(
        default_factory=lambda: os.getenv("BACKEND_URL", "http://localhost:8000")
    )

    # --- Camera settings ---
    camera_fps: int = field(
        default_factory=lambda: int(os.getenv("CAMERA_FPS", "15"))
    )
    camera_resolution: tuple[int, int] = (640, 480)
    use_depth: bool = True

    # --- Pose estimation ---
    confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))
    )
    model_complexity: int = 1  # 0=lite, 1=full, 2=heavy

    # --- Exercise classification ---
    rep_cooldown_frames: int = 10  # Minimum frames between rep counts
    angle_smoothing_window: int = 5  # Moving average window for angle smoothing

    # --- Display ---
    show_preview: bool = field(
        default_factory=lambda: os.getenv("SHOW_PREVIEW", "true").lower() == "true"
    )
    preview_width: int = 960
    preview_height: int = 720


config = EdgeConfig()
