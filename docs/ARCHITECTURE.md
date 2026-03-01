# Architecture

## System Overview

FitForm AI follows an edge-cloud hybrid architecture. Compute-intensive pose estimation runs on the edge device (Jetson Orin Nano) to minimize latency, while AI coaching and data persistence live in Azure.

## Component Diagram

```
┌─────────────────────────────────────────────┐
│              EDGE LAYER                      │
│         Jetson Orin Nano (8GB)              │
│                                             │
│  OAK-D Lite ──→ DepthAI SDK               │
│       │                                     │
│       ▼                                     │
│  MediaPipe BlazePose (33 landmarks)        │
│       │                                     │
│       ├──→ ROMCalculator (joint angles)    │
│       │                                     │
│       └──→ ExerciseClassifier              │
│              (rule-based + rep counting)    │
│              │                              │
│              ▼                              │
│         AzureClient (HTTP POST)            │
└─────────────┬───────────────────────────────┘
              │ REST API (JSON)
              ▼
┌─────────────────────────────────────────────┐
│             CLOUD LAYER                      │
│         Azure App Service (B1)              │
│                                             │
│  FastAPI ──→ SessionStore (in-memory)      │
│       │                                     │
│       ├──→ SSE Stream ──→ Dashboard        │
│       │                                     │
│       └──→ AICoach ──→ Azure OpenAI        │
│                         (GPT-4o)            │
└─────────────────────────────────────────────┘
```

## Data Flow

1. **Frame Capture**: OAK-D Lite captures RGB + depth at 15 FPS via DepthAI SDK
2. **Pose Estimation**: MediaPipe BlazePose extracts 33 body landmarks per frame
3. **Angle Computation**: ROMCalculator computes 8 joint angles with moving-average smoothing
4. **Classification**: ExerciseClassifier detects exercise type and counts reps using hysteresis thresholds
5. **Telemetry Upload**: Every 5th frame's data is POSTed to the backend API
6. **Real-Time Dashboard**: Backend broadcasts via SSE; browser renders Chart.js visualizations
7. **AI Coaching**: On-demand, session data is sent to Azure OpenAI for form analysis

## Key Design Decisions

| Decision | Rationale |
|---|---|
| MediaPipe over custom model | Pre-trained, GPU-accelerated, 33-point accuracy sufficient for ROM |
| Rule-based classifier over ML | Interpretable, no training data needed, easily extensible |
| SSE over WebSocket | Simpler server implementation, unidirectional is sufficient |
| In-memory store over DB | MVP speed; production path is Azure Cosmos DB |
| Edge-first pose estimation | Sub-100ms latency; no cloud dependency for core function |

## Security Considerations

- API keys stored in environment variables, never in code
- CORS restricted in production deployment
- No PII stored; telemetry is anonymous joint angles
- Azure OpenAI accessed via managed identity in production

## Scalability Path

| Current (MVP) | Production |
|---|---|
| In-memory session store | Azure Cosmos DB |
| Single backend instance | Azure Container Apps (auto-scale) |
| Manual deployment | GitHub Actions → Azure deployment |
| Rule-based classifier | Fine-tuned classification model via Azure ML |
| Single camera | Multi-camera orchestration |
