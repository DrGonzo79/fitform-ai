# FitForm AI — Real-Time Exercise Recognition & Range of Motion Analysis

[![CI/CD](https://github.com/YOUR_USERNAME/fitform-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/fitform-ai/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Azure AI Foundry](https://img.shields.io/badge/Azure-AI%20Foundry-0078D4)](https://ai.azure.com)

> Edge-to-cloud fitness analysis platform combining on-device pose estimation with Azure AI–powered coaching feedback. Built for real-time exercise classification and biomechanical range-of-motion tracking.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

FitForm AI is a full-stack application that uses computer vision to recognize exercises in real time and measure biomechanical range of motion. The system runs pose estimation on an NVIDIA Jetson Orin Nano with a Luxonis OAK-D Lite depth camera, classifies movements using joint-angle heuristics, and streams telemetry to an Azure-hosted backend. An Azure OpenAI integration provides AI-generated coaching feedback based on observed form.

**Use Cases:**
- Personal training form analysis
- Physical therapy ROM tracking
- Corporate wellness program analytics
- Remote coaching and athlete monitoring

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                   EDGE DEVICE                        │
│              Jetson Orin Nano + OAK-D Lite            │
│                                                      │
│  ┌─────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │ DepthAI │→│ MediaPipe    │→│ Exercise       │  │
│  │ Camera  │  │ Pose (33 pt) │  │ Classifier     │  │
│  └─────────┘  └──────────────┘  └───────┬────────┘  │
│                                         │            │
│                              ┌──────────▼─────────┐  │
│                              │ ROM Calculator     │  │
│                              │ (Joint Angles)     │  │
│                              └──────────┬─────────┘  │
│                                         │ REST API   │
└─────────────────────────────────────────┼────────────┘
                                          │
                              ┌───────────▼────────────┐
                              │    AZURE CLOUD          │
                              │                         │
                              │  ┌───────────────────┐  │
                              │  │ FastAPI Backend    │  │
                              │  │ (App Service)     │  │
                              │  └─────┬─────────────┘  │
                              │        │                │
                              │  ┌─────▼─────────────┐  │
                              │  │ Azure OpenAI      │  │
                              │  │ (GPT-4o Coach)    │  │
                              │  └───────────────────┘  │
                              │                         │
                              │  ┌───────────────────┐  │
                              │  │ Static Web App    │  │
                              │  │ (Dashboard)       │  │
                              │  └───────────────────┘  │
                              └─────────────────────────┘
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed component documentation.

---

## Features

| Feature | Description |
|---|---|
| **Real-Time Pose Estimation** | 33-point body landmark detection via MediaPipe BlazePose at 15–30 FPS on Jetson |
| **Exercise Classification** | Rule-based classifier for air squats, push-ups, sit-ups with rep counting |
| **Range of Motion Tracking** | Joint angle calculation (knee, hip, elbow, shoulder) with min/max per rep |
| **Depth-Enhanced Analysis** | OAK-D Lite stereo depth for 3D spatial awareness and distance calibration |
| **AI Coaching Feedback** | Azure OpenAI (GPT-4o) generates personalized form corrections per session |
| **Live Dashboard** | Browser-based real-time visualization of exercise metrics and ROM data |
| **Session Persistence** | Exercise sessions stored with full telemetry for longitudinal analysis |
| **REST API** | Documented FastAPI endpoints for integration with external systems |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Edge Hardware** | NVIDIA Jetson Orin Nano (8GB), Luxonis OAK-D Lite |
| **Pose Estimation** | MediaPipe BlazePose (GPU-accelerated) |
| **Camera SDK** | DepthAI 2.x |
| **Edge Runtime** | Python 3.10, OpenCV, NumPy |
| **Backend** | FastAPI, Uvicorn, Pydantic |
| **AI Services** | Azure OpenAI (GPT-4o) via Azure AI Foundry |
| **Frontend** | HTML5, Vanilla JS, Chart.js, Server-Sent Events |
| **Infrastructure** | Azure App Service, Azure Static Web Apps |
| **CI/CD** | GitHub Actions |
| **Testing** | pytest, httpx |

---

## Quick Start

### Prerequisites

- NVIDIA Jetson Orin Nano with JetPack 6.x
- Luxonis OAK-D Lite camera (USB-C connected)
- Azure subscription with Azure OpenAI resource provisioned
- Python 3.10+
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/fitform-ai.git
cd fitform-ai
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your Azure credentials
```

### 3. Start the Backend (Azure or local)

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 4. Start the Edge Client (on Jetson)

```bash
cd edge
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

### 5. Open the Dashboard

Navigate to `http://<backend-ip>:8000` in your browser.

See [docs/SETUP.md](docs/SETUP.md) for detailed installation instructions.

---

## Project Structure

```
fitform-ai/
├── edge/                          # Jetson edge application
│   ├── main.py                    # Entry point — camera loop + orchestration
│   ├── config.py                  # Edge configuration management
│   ├── pose_estimator.py          # MediaPipe pose landmark extraction
│   ├── exercise_classifier.py     # Rule-based exercise detection + rep counting
│   ├── rom_calculator.py          # Joint angle & range of motion math
│   ├── azure_client.py            # Backend API client
│   └── requirements.txt
├── backend/                       # Azure-hosted FastAPI backend
│   ├── main.py                    # App factory + middleware + static serving
│   ├── models.py                  # Pydantic schemas
│   ├── routers/
│   │   ├── exercises.py           # Exercise telemetry endpoints
│   │   └── sessions.py           # Session management endpoints
│   ├── services/
│   │   ├── ai_coach.py           # Azure OpenAI coaching integration
│   │   └── analytics.py          # Session analytics & aggregation
│   └── requirements.txt
├── frontend/
│   └── index.html                 # Single-page dashboard (SSE + Chart.js)
├── tests/                         # Test suite
│   ├── test_classifier.py
│   ├── test_rom.py
│   └── test_api.py
├── docs/                          # Extended documentation
│   ├── ARCHITECTURE.md
│   ├── SETUP.md
│   ├── API.md
│   └── DEPLOYMENT.md
├── .github/workflows/ci.yml      # GitHub Actions CI pipeline
├── .env.example                   # Environment variable template
├── .gitignore
├── Dockerfile.backend             # Backend container image
├── CONTRIBUTING.md
├── SECURITY.md
└── LICENSE
```

---

## Configuration

All configuration is managed through environment variables. Copy `.env.example` to `.env` and populate:

| Variable | Description | Required |
|---|---|---|
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI resource endpoint | Yes |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | Yes |
| `AZURE_OPENAI_DEPLOYMENT` | Model deployment name (e.g., `gpt-4o`) | Yes |
| `AZURE_OPENAI_API_VERSION` | API version (default: `2024-12-01-preview`) | No |
| `BACKEND_URL` | Backend URL for edge client | Yes (edge) |
| `CAMERA_FPS` | Target camera FPS (default: 15) | No |
| `CONFIDENCE_THRESHOLD` | Pose detection confidence (default: 0.6) | No |

---

## API Reference

See [docs/API.md](docs/API.md) for complete endpoint documentation.

**Key Endpoints:**

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/exercises/frame` | Submit a single frame's exercise telemetry |
| `GET` | `/api/v1/sessions` | List all exercise sessions |
| `GET` | `/api/v1/sessions/{id}` | Get session detail with ROM analytics |
| `POST` | `/api/v1/sessions/{id}/coach` | Generate AI coaching feedback for a session |
| `GET` | `/api/v1/stream` | SSE stream for real-time dashboard updates |

---

## Testing

```bash
cd tests
pip install -r ../backend/requirements.txt
pytest -v --tb=short
```

See test coverage targets in [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Deployment

### Azure App Service (Backend)

```bash
az webapp up --name fitform-ai-api \
  --resource-group fitform-rg \
  --runtime "PYTHON:3.11" \
  --sku B1
```

### Azure Static Web Apps (Dashboard)

```bash
az staticwebapp create \
  --name fitform-ai-dashboard \
  --resource-group fitform-rg \
  --source frontend/
```

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for full deployment guide including CI/CD setup.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines, branching strategy, and code review process.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [MediaPipe](https://google.github.io/mediapipe/) — Google's on-device ML framework
- [DepthAI](https://docs.luxonis.com/) — Luxonis camera SDK
- [Azure AI Foundry](https://ai.azure.com) — Microsoft's AI platform
- Built by [Justin YOUR_LAST_NAME](https://github.com/YOUR_USERNAME) as a demonstration of full-stack AI/ML engineering
