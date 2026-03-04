"""
Introspect DepthAI v3 Camera + Device API to determine the correct
initialization sequence.

Usage:
    python edge/inspect_v3_api.py
"""

import depthai as dai
import inspect

print(f"DepthAI version: {dai.__version__}")
print()

# 1. Check Camera.build() signature
cam_cls = dai.node.Camera
print("=== Camera.build() signature ===")
try:
    build_method = getattr(cam_cls, "build", None)
    if build_method:
        sig = inspect.signature(build_method)
        print(f"  build{sig}")
        print(f"  docstring: {build_method.__doc__}")
    else:
        print("  NO build method found")
except Exception as e:
    print(f"  Error inspecting build: {e}")

print()

# 2. Check requestOutput signature
print("=== Camera.requestOutput() signature ===")
try:
    req_method = getattr(cam_cls, "requestOutput", None)
    if req_method:
        sig = inspect.signature(req_method)
        print(f"  requestOutput{sig}")
        print(f"  docstring: {req_method.__doc__}")
    else:
        print("  NO requestOutput method found")
except Exception as e:
    print(f"  Error inspecting requestOutput: {e}")

print()

# 3. Check if Pipeline has a start/build method
print("=== Pipeline methods (start/build/run) ===")
p_methods = [m for m in dir(dai.Pipeline) if not m.startswith("_")]
for m in p_methods:
    if any(k in m.lower() for k in ["start", "build", "run", "init", "create"]):
        print(f"  {m}")
print(f"  All pipeline methods: {p_methods}")

print()

# 4. Check Device constructor signatures
print("=== Device constructor / init ===")
try:
    sig = inspect.signature(dai.Device.__init__)
    print(f"  Device.__init__{sig}")
except Exception as e:
    print(f"  Error: {e}")

print()

# 5. Try approach A: build() inside Device context
print("=== Approach A: pipeline -> Device -> build -> requestOutput ===")
try:
    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.Camera)
    with dai.Device(pipeline) as device:
        cam.build()
        q = cam.requestOutput((640, 480), type=dai.ImgFrame.Type.BGR888p)
        frame = q.get().getCvFrame()
        print(f"  SUCCESS! Frame shape: {frame.shape}")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")

print()

# 6. Try approach B: Device() without pipeline, then create nodes
print("=== Approach B: Device() -> create -> build -> requestOutput ===")
try:
    with dai.Device() as device:
        pipeline = device.getPipeline()
        cam = pipeline.create(dai.node.Camera)
        cam.build()
        q = cam.requestOutput((640, 480), type=dai.ImgFrame.Type.BGR888p)
        frame = q.get().getCvFrame()
        print(f"  SUCCESS! Frame shape: {frame.shape}")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")

print()

# 7. Try approach C: Pipeline -> build -> requestOutput -> Device
print("=== Approach C: pipeline -> build -> requestOutput -> Device (current code) ===")
try:
    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.Camera)
    cam.build()
    q = cam.requestOutput((640, 480), type=dai.ImgFrame.Type.BGR888p)
    with dai.Device(pipeline) as device:
        frame = q.get().getCvFrame()
        print(f"  SUCCESS! Frame shape: {frame.shape}")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")

print()

# 8. Try approach D: Device() no-arg, create camera directly
print("=== Approach D: Device() no pipeline arg ===")
try:
    dev_methods = [m for m in dir(dai.Device) if not m.startswith("_") and "create" in m.lower()]
    print(f"  Device create methods: {dev_methods}")
    pipe_methods = [m for m in dir(dai.Device) if not m.startswith("_") and "pipe" in m.lower()]
    print(f"  Device pipeline methods: {pipe_methods}")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")

print()
print("=== Done ===")
