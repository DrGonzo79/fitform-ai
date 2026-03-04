"""
DepthAI v3 API — find the correct pipeline-driven initialization.

In v3, Device() no longer accepts a pipeline. The Pipeline itself
manages device connection via build()/start()/run().

Each approach runs in a subprocess so a leaked device connection
can't poison later tests.

Usage:
    python edge/inspect_v3_api.py
"""

import subprocess
import sys
import textwrap

APPROACHES = {
    "A": textwrap.dedent("""\
        # Pipeline.start() approach
        import depthai as dai, time
        pipeline = dai.Pipeline()
        cam = pipeline.create(dai.node.Camera)
        cam.build()
        q = cam.requestOutput((640, 480), type=dai.ImgFrame.Type.BGR888p)
        pipeline.start()
        frame = q.get().getCvFrame()
        print(f"SUCCESS! Frame shape: {frame.shape}")
        pipeline.stop()
    """),

    "B": textwrap.dedent("""\
        # Pipeline.start() — build node AFTER pipeline.start()
        import depthai as dai, time
        pipeline = dai.Pipeline()
        cam = pipeline.create(dai.node.Camera)
        pipeline.start()
        cam.build()
        q = cam.requestOutput((640, 480), type=dai.ImgFrame.Type.BGR888p)
        frame = q.get().getCvFrame()
        print(f"SUCCESS! Frame shape: {frame.shape}")
        pipeline.stop()
    """),

    "C": textwrap.dedent("""\
        # Device() then pipeline.start(device)
        import depthai as dai, time
        pipeline = dai.Pipeline()
        cam = pipeline.create(dai.node.Camera)
        cam.build()
        q = cam.requestOutput((640, 480), type=dai.ImgFrame.Type.BGR888p)
        with dai.Device() as device:
            pipeline.start(device)
            frame = q.get().getCvFrame()
            print(f"SUCCESS! Frame shape: {frame.shape}")
            pipeline.stop()
    """),

    "D": textwrap.dedent("""\
        # Device() then pipeline.start() (no device arg)
        import depthai as dai, time
        with dai.Device() as device:
            pipeline = dai.Pipeline()
            cam = pipeline.create(dai.node.Camera)
            cam.build()
            q = cam.requestOutput((640, 480), type=dai.ImgFrame.Type.BGR888p)
            pipeline.start()
            frame = q.get().getCvFrame()
            print(f"SUCCESS! Frame shape: {frame.shape}")
            pipeline.stop()
    """),

    "E": textwrap.dedent("""\
        # pipeline.run() instead of start (blocks, so use timeout)
        import depthai as dai, time, threading
        pipeline = dai.Pipeline()
        cam = pipeline.create(dai.node.Camera)
        cam.build()
        q = cam.requestOutput((640, 480), type=dai.ImgFrame.Type.BGR888p)
        t = threading.Thread(target=pipeline.run, daemon=True)
        t.start()
        time.sleep(2)
        frame = q.get().getCvFrame()
        print(f"SUCCESS! Frame shape: {frame.shape}")
        pipeline.stop()
    """),

    "F": textwrap.dedent("""\
        # Inspect pipeline.start() signature and try passing device
        import depthai as dai
        print("pipeline.start doc:", dai.Pipeline.start.__doc__)
        print("pipeline.build doc:", dai.Pipeline.build.__doc__)
        print("pipeline.run doc:", dai.Pipeline.run.__doc__)
    """),
}


def run_approach(label: str, code: str) -> None:
    print(f"=== Approach {label} ===")
    for line in code.strip().splitlines()[:2]:
        print(f"  {line.strip()}")
    print()

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=15,
    )

    if result.stdout.strip():
        print(f"  stdout: {result.stdout.strip()}")
    if result.stderr.strip():
        # Only show last 5 lines of stderr to keep it readable
        err_lines = result.stderr.strip().splitlines()
        for line in err_lines[-5:]:
            print(f"  stderr: {line}")
    print()


if __name__ == "__main__":
    import depthai as dai
    print(f"DepthAI version: {dai.__version__}")
    print()

    # Run F first (docs inspection, no device needed)
    run_approach("F", APPROACHES["F"])

    # Then try each device approach in isolation
    for label in ["A", "B", "C", "D", "E"]:
        try:
            run_approach(label, APPROACHES[label])
        except subprocess.TimeoutExpired:
            print(f"  TIMED OUT (15s)\n")
