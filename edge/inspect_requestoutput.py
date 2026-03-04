"""Inspect requestOutput signature — paste the output back to Copilot."""

import depthai as dai

p = dai.Pipeline()
cam = p.create(dai.node.Camera)

help(cam.requestOutput)
