"""
OAK-D Lite camera connectivity test.

Detects the installed DepthAI version and reads 30 frames
to confirm the camera is working.

Usage:
    python edge/test_oakd.py
"""

import time

import depthai as dai

HAS_XLINK = hasattr(dai.node, "XLinkOut")

print(f"DepthAI version: {dai.__version__}")
print(f"API: {'v2 (XLinkOut)' if HAS_XLINK else 'v3 (requestOutput)'}")

FRAMES = 30

if HAS_XLINK:
    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(640, 480)
    cam.setInterleaved(False)
    cam.setFps(30)
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("preview")
    cam.preview.link(xout.input)
    with dai.Device(pipeline) as device:
        q = device.getOutputQueue("preview", maxSize=4, blocking=False)
        print(f"OAK-D Lite streaming! Reading {FRAMES} frames...")
        for i in range(FRAMES):
            frame = q.get().getCvFrame()
            print(f"  Frame {i + 1}: {frame.shape}")
            time.sleep(0.03)
        print("Camera test PASSED!")
else:
    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.Camera)
    q = cam.requestOutput((640, 480), type=dai.ImgFrame.Type.BGR888p)
    with dai.Device(pipeline) as device:
        print(f"Device: {device.getDeviceName()}")
        print(f"OAK-D Lite streaming! Reading {FRAMES} frames...")
        for i in range(FRAMES):
            frame = q.get().getCvFrame()
            print(f"  Frame {i + 1}: {frame.shape}")
            time.sleep(0.03)
        print("Camera test PASSED!")
