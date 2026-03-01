# Setup Guide

## Prerequisites

| Requirement | Version | Purpose |
|---|---|---|
| NVIDIA Jetson Orin Nano | JetPack 6.x | Edge compute |
| Luxonis OAK-D Lite | Firmware latest | Camera + depth |
| Python | 3.10+ | Runtime |
| Azure Subscription | — | OpenAI + hosting |
| Git | 2.x+ | Version control |

## Jetson Orin Nano Setup

### 1. Flash JetPack

Follow NVIDIA's [JetPack installation guide](https://developer.nvidia.com/embedded/jetpack). Ensure CUDA and cuDNN are installed.

### 2. Install DepthAI

```bash
sudo apt update && sudo apt install -y python3-pip libusb-1.0-0-dev
pip3 install depthai
```

Verify camera connection:
```bash
python3 -c "import depthai; print(depthai.Device.getAllAvailableDevices())"
```

### 3. Install Edge Dependencies

```bash
cd fitform-ai/edge
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Azure AI Foundry Setup

### 1. Create Azure OpenAI Resource

1. Go to [Azure AI Foundry](https://ai.azure.com)
2. Create a new project
3. Deploy a `gpt-4o` model
4. Copy the endpoint URL and API key

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env`:
```
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_DEPLOYMENT=gpt-4o
```

## Running Locally

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Verify: `curl http://localhost:8000/health`

### Edge Client

```bash
cd edge
source venv/bin/activate
python main.py                # OAK-D Lite
python main.py --webcam       # USB webcam fallback
python main.py --no-upload    # Local only
```

### Dashboard

Open `http://localhost:8000` in a browser.

## Troubleshooting

| Issue | Solution |
|---|---|
| `depthai` import error | Ensure USB-C cable is data-capable; run `sudo udevadm control --reload-rules` |
| MediaPipe GPU errors | Install `mediapipe` with GPU support: `pip install mediapipe-gpu` |
| Low FPS | Reduce `CAMERA_FPS` to 10 or set `model_complexity=0` in config |
| Backend connection refused | Check `BACKEND_URL` matches the server's IP and port |
