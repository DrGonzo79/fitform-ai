"""DepthAI Camera node introspection — paste the output back to Copilot."""

import depthai as dai

p = dai.Pipeline()
cam = p.create(dai.node.Camera)

print("All Camera methods/attrs:")
for m in sorted(dir(cam)):
    if not m.startswith("_"):
        print(f"  {m}")
