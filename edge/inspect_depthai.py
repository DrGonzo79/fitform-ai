"""DepthAI API introspection — paste the output back to Copilot."""

import depthai as dai

print(f"depthai version: {dai.__version__}")

p = dai.Pipeline()
cam = p.create(dai.node.Camera)

print("Camera methods:", [m for m in dir(cam) if not m.startswith("_") and "out" in m.lower()])
print("Pipeline methods:", [m for m in dir(p) if not m.startswith("_") and ("out" in m.lower() or "queue" in m.lower() or "link" in m.lower())])
print("Device methods:", [m for m in dir(dai.Device) if not m.startswith("_") and ("out" in m.lower() or "queue" in m.lower())])
