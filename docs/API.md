# API Reference

Base URL: `http://<host>:8000`

Interactive docs available at `/docs` (Swagger UI) and `/redoc` (ReDoc).

## Endpoints

### Health Check

```
GET /health
```

Response:
```json
{
    "status": "healthy",
    "service": "fitform-ai-backend",
    "azure_openai": true,
    "active_sessions": 2
}
```

### Sessions

#### Create Session
```
POST /api/v1/sessions
Content-Type: application/json

{"started_at": 1700000000}
```

Response: `SessionResponse`

#### List Sessions
```
GET /api/v1/sessions
```

Response: `SessionResponse[]`

#### Get Session
```
GET /api/v1/sessions/{session_id}
```

Response: `SessionResponse`

#### Update Session
```
PATCH /api/v1/sessions/{session_id}
Content-Type: application/json

{"ended_at": 1700003600}
```

### Exercise Telemetry

#### Submit Frame
```
POST /api/v1/exercises/frame
Content-Type: application/json

{
    "session_id": "a1b2c3d4",
    "timestamp": 1700000001.5,
    "exercise": "air_squat",
    "rep_count": 5,
    "phase": "up",
    "confidence": 0.87,
    "angles": {
        "left_knee": 85.2,
        "right_knee": 87.1,
        "left_hip": 82.0,
        "right_hip": 83.5
    },
    "rom_summary": {
        "left_knee": {"min": 78.0, "max": 172.0, "range": 94.0}
    }
}
```

#### Live Stream (SSE)
```
GET /api/v1/exercises/stream
Accept: text/event-stream
```

Events:
```
data: {"session_id": "a1b2c3d4", "exercise": "air_squat", "rep_count": 5, ...}
```

### AI Coaching

#### Generate Feedback
```
POST /api/v1/sessions/{session_id}/coach
```

Response:
```json
{
    "session_id": "a1b2c3d4",
    "feedback": "Your squat depth is excellent...",
    "form_score": 7.5,
    "recommendations": [
        "Focus on keeping knees tracking over toes",
        "Maintain upright torso position"
    ],
    "model": "gpt-4o"
}
```

## Data Models

### ExerciseType (enum)
`air_squat` | `push_up` | `sit_up` | `unknown`

### Phase (enum)
`up` | `down` | `neutral`
