"""
FitForm AI — Backend API Server

FastAPI application serving exercise telemetry endpoints,
session management, AI coaching, and the live dashboard.

Designed for deployment on Azure App Service or Azure Container Apps.

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from dependencies import limiter
from routers import exercises, sessions
from services.ai_coach import AICoach
from services.analytics import SessionStore

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fitform.backend")

# ---------------------------------------------------------------------------
# Singleton Services
# ---------------------------------------------------------------------------

session_store = SessionStore()
ai_coach = AICoach()


# ---------------------------------------------------------------------------
# App Factory
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FitForm AI Backend starting...")
    logger.info(
        "Azure OpenAI: %s",
        "configured" if os.getenv("AZURE_OPENAI_ENDPOINT") else "NOT configured (mock mode)",
    )
    yield
    logger.info("FitForm AI Backend shutting down.")


app = FastAPI(
    title="FitForm AI",
    description="Real-time exercise recognition and range-of-motion analysis API",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(exercises.router)
app.include_router(sessions.router)

# ---------------------------------------------------------------------------
# Static Files & Dashboard
# ---------------------------------------------------------------------------

frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


@app.get("/")
async def dashboard():
    """Serve the live dashboard."""
    index_path = frontend_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse(
        {"message": "FitForm AI API", "docs": "/docs"},
        status_code=200,
    )


@app.get("/health")
async def health_check():
    """Health check endpoint for Azure App Service."""
    return {
        "status": "healthy",
        "service": "fitform-ai-backend",
        "azure_openai": bool(os.getenv("AZURE_OPENAI_ENDPOINT")),
        "active_sessions": len(session_store.list_sessions()),
    }
