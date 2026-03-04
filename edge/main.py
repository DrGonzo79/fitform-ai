"""
FitForm AI — Edge Application Entry Point

Real-time exercise recognition and range-of-motion analysis running
on NVIDIA Jetson Orin Nano with Luxonis OAK-D Lite camera.

Pipeline:
    Camera Frame → Pose Estimation → Exercise Classification → ROM Calculation → Backend Upload

Usage:
    python main.py                    # Run with OAK-D Lite
    python main.py --webcam           # Fallback to USB webcam
    python main.py --no-upload        # Local only (no backend)

Controls:
    q / ESC  — Quit
    r        — Reset rep counters
    c        — Request coaching feedback
    s        — Print session summary
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Optional

import cv2
import numpy as np

from azure_client import AzureClient
from config import config
from exercise_classifier import ClassificationResult, ExerciseClassifier, ExerciseType
from pose_estimator import PoseEstimator
from rom_calculator import ROMCalculator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fitform.edge")


# ---------------------------------------------------------------------------
# Camera Backends
# ---------------------------------------------------------------------------

def create_oakd_camera():
    """
    Initialize OAK-D Lite camera via DepthAI SDK.

    Returns a generator that yields (bgr_frame, depth_frame) tuples.
    Falls back to webcam if DepthAI is not available.

    Supports both DepthAI v2 (XLinkOut nodes) and v3 (requestOutput API).
    """
    try:
        import depthai as dai

        _has_xlink = hasattr(dai.node, "XLinkOut")

        if _has_xlink:
            yield from _oakd_v2(dai)
        else:
            yield from _oakd_v3(dai)

    except (ImportError, RuntimeError) as e:
        logger.warning("DepthAI not available (%s), falling back to webcam", e)
        yield from create_webcam_camera()


def _oakd_v2(dai):
    """DepthAI v2.x pipeline using XLinkOut nodes."""
    pipeline = dai.Pipeline()

    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(config.camera_resolution[0], config.camera_resolution[1])
    cam_rgb.setInterleaved(False)
    cam_rgb.setFps(config.camera_fps)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    if config.use_depth:
        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_right = pipeline.create(dai.node.MonoCamera)
        stereo = pipeline.create(dai.node.StereoDepth)

        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setCamera("left")
        mono_right.setCamera("right")

        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)

        xout_depth = pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")
        stereo.depth.link(xout_depth.input)

    with dai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
        q_depth = device.getOutputQueue("depth", maxSize=4, blocking=False) if config.use_depth else None

        logger.info("OAK-D Lite camera initialized (DepthAI v2 API)")

        while True:
            in_rgb = q_rgb.get()
            frame = in_rgb.getCvFrame()

            depth_frame = None
            if q_depth:
                in_depth = q_depth.tryGet()
                if in_depth:
                    depth_frame = in_depth.getFrame()

            yield frame, depth_frame


def _oakd_v3(dai):
    """DepthAI v3.x pipeline using requestOutput API (no XLinkOut)."""
    w, h = config.camera_resolution

    pipeline = dai.Pipeline()

    # RGB camera
    cam = pipeline.create(dai.node.Camera)
    cam.build()
    q_rgb = cam.requestOutput((w, h), type=dai.ImgFrame.Type.BGR888p)

    # Stereo depth
    q_depth = None
    if config.use_depth:
        try:
            stereo = pipeline.create(dai.node.StereoDepth)
            q_depth = stereo.requestOutput((w, h))
        except Exception as e:
            logger.warning("Stereo depth not available (%s), continuing RGB-only", e)

    with dai.Device(pipeline) as device:
        logger.info("OAK-D Lite camera initialized (DepthAI v3 API)")

        while device.isRunning():
            in_rgb = q_rgb.get()
            frame = in_rgb.getCvFrame()

            depth_frame = None
            if q_depth:
                in_depth = q_depth.tryGet()
                if in_depth:
                    depth_frame = in_depth.getCvFrame()

            yield frame, depth_frame


def create_webcam_camera():
    """Fallback: standard USB webcam via OpenCV."""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera_resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera_resolution[1])
    cap.set(cv2.CAP_PROP_FPS, config.camera_fps)

    if not cap.isOpened():
        logger.error("Cannot open webcam")
        sys.exit(1)

    logger.info("Webcam initialized")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame, None
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# HUD Overlay
# ---------------------------------------------------------------------------

def draw_hud(
    frame: np.ndarray,
    result: Optional[ClassificationResult],
    fps: float,
    session_id: Optional[str],
) -> np.ndarray:
    """Draw heads-up display with exercise info, rep count, and angles."""
    h, w = frame.shape[:2]

    # Semi-transparent overlay bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.0f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if result:
        # Exercise name
        label = result.exercise.value.replace("_", " ").upper()
        color = (0, 255, 255) if result.exercise != ExerciseType.UNKNOWN else (128, 128, 128)
        cv2.putText(frame, label, (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Rep count
        cv2.putText(frame, f"REPS: {result.rep_count}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Phase indicator
        phase_color = {
            "up": (0, 255, 0),
            "down": (0, 0, 255),
            "neutral": (128, 128, 128),
        }
        cv2.putText(frame, result.phase.value.upper(), (250, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, phase_color.get(result.phase.value, (255, 255, 255)), 2)

        # Key angles on right side
        x_offset = w - 280
        key_angles = _get_key_angles(result)
        for i, (name, val) in enumerate(key_angles.items()):
            cv2.putText(frame, f"{name}: {val:.0f}°", (x_offset, 25 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Session ID
    if session_id:
        cv2.putText(frame, f"Session: {session_id[:12]}...", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    return frame


def _get_key_angles(result: ClassificationResult) -> dict[str, float]:
    """Return the most relevant angles for the current exercise."""
    angles = result.angles
    if result.exercise == ExerciseType.AIR_SQUAT:
        return {"L Knee": angles["left_knee"], "R Knee": angles["right_knee"],
                "L Hip": angles["left_hip"], "R Hip": angles["right_hip"]}
    elif result.exercise == ExerciseType.PUSH_UP:
        return {"L Elbow": angles["left_elbow"], "R Elbow": angles["right_elbow"],
                "L Shoulder": angles["left_shoulder"], "R Shoulder": angles["right_shoulder"]}
    elif result.exercise == ExerciseType.SIT_UP:
        return {"L Hip": angles["left_hip"], "R Hip": angles["right_hip"],
                "L Knee": angles["left_knee"], "R Knee": angles["right_knee"]}
    return dict(list(angles.items())[:4])


# ---------------------------------------------------------------------------
# Main Loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="FitForm AI Edge Client")
    parser.add_argument("--webcam", action="store_true", help="Use USB webcam instead of OAK-D Lite")
    parser.add_argument("--no-upload", action="store_true", help="Disable backend telemetry upload")
    parser.add_argument("--no-display", action="store_true", help="Headless mode (no preview window)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("FitForm AI — Edge Client Starting")
    logger.info("Backend: %s", config.backend_url)
    logger.info("=" * 60)

    # Initialize components
    pose = PoseEstimator(config)
    rom = ROMCalculator(smoothing_window=config.angle_smoothing_window)
    classifier = ExerciseClassifier(cooldown_frames=config.rep_cooldown_frames)

    client: Optional[AzureClient] = None
    if not args.no_upload:
        client = AzureClient(config.backend_url)
        client.create_session()

    # Camera source
    camera = create_webcam_camera() if args.webcam else create_oakd_camera()

    frame_count = 0
    fps_timer = time.time()
    fps = 0.0
    upload_interval = 5  # Send telemetry every N frames
    last_result: Optional[ClassificationResult] = None

    try:
        for frame, depth_frame in camera:
            frame_count += 1

            # --- Pose estimation ---
            pose_result = pose.process_frame(frame)

            classification = None
            if pose_result.detected:
                # --- Joint angles ---
                angles = rom.compute_angles(pose_result.landmark_array)

                # --- Exercise classification ---
                classification = classifier.classify(
                    pose_result.landmark_array, angles
                )
                last_result = classification

                # --- Annotate frame ---
                frame = pose.draw_landmarks(frame, pose_result)

            # --- FPS calculation ---
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_timer = time.time()

            # --- Upload telemetry ---
            if client and classification and frame_count % upload_interval == 0:
                telemetry = {
                    "exercise": classification.exercise.value,
                    "rep_count": classification.rep_count,
                    "phase": classification.phase.value,
                    "confidence": classification.confidence,
                    "angles": classification.angles,
                    "rom_summary": rom.get_rom_summary(),
                }
                client.send_frame(telemetry)

            # --- Display ---
            if not args.no_display and config.show_preview:
                display = draw_hud(
                    frame, classification, fps,
                    client.session_id if client else None,
                )
                display = cv2.resize(display, (config.preview_width, config.preview_height))
                cv2.imshow("FitForm AI", display)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # q or ESC
                    break
                elif key == ord("r"):
                    classifier.reset()
                    rom.reset()
                    logger.info("Counters reset")
                elif key == ord("c") and client:
                    logger.info("Requesting coaching feedback...")
                    coaching = client.get_coaching()
                    if coaching:
                        logger.info("Coach: %s", coaching.get("feedback", ""))
                elif key == ord("s"):
                    _print_summary(classifier, rom)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        _print_summary(classifier, rom)
        if client:
            client.end_session()
            client.close()
        pose.close()
        cv2.destroyAllWindows()
        logger.info("Shutdown complete.")


def _print_summary(classifier: ExerciseClassifier, rom: ROMCalculator) -> None:
    """Print session summary to console."""
    logger.info("=" * 40)
    logger.info("SESSION SUMMARY")
    logger.info("-" * 40)
    for exercise, count in classifier.get_all_counts().items():
        if count > 0:
            logger.info("  %s: %d reps", exercise.replace("_", " ").title(), count)
    logger.info("-" * 40)
    for joint, stats in rom.get_rom_summary().items():
        if stats["range"] > 5:  # Only show joints that moved
            logger.info("  %s ROM: %.0f° - %.0f° (range: %.0f°)",
                        joint.replace("_", " ").title(),
                        stats["min"], stats["max"], stats["range"])
    logger.info("=" * 40)


if __name__ == "__main__":
    main()
