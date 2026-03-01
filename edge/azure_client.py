"""
Azure backend API client.

Handles communication between the edge device and the Azure-hosted
FastAPI backend. Sends exercise telemetry frames and manages session
lifecycle. Uses async HTTP with connection pooling for low latency.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class AzureClient:
    """
    HTTP client for the FitForm AI backend API.

    Manages session creation, frame telemetry submission, and
    coaching feedback retrieval. Includes retry logic and
    connection pooling for reliable edge-to-cloud communication.

    Args:
        base_url: Backend API base URL.
        timeout: Request timeout in seconds.
    """

    def __init__(self, base_url: str, timeout: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session_id: Optional[str] = None

        # Configure connection pooling and retry
        self._http = requests.Session()
        retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503])
        adapter = HTTPAdapter(max_retries=retry, pool_maxsize=4)
        self._http.mount("http://", adapter)
        self._http.mount("https://", adapter)

        logger.info("AzureClient initialized | backend=%s", self.base_url)

    def create_session(self) -> str:
        """
        Create a new exercise session on the backend.

        Returns:
            Session ID string.
        """
        try:
            resp = self._http.post(
                f"{self.base_url}/api/v1/sessions",
                json={"started_at": time.time()},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            self._session_id = data["session_id"]
            logger.info("Session created: %s", self._session_id)
            return self._session_id
        except requests.RequestException as e:
            logger.warning("Failed to create session: %s", e)
            # Generate local session ID as fallback
            self._session_id = f"local-{int(time.time())}"
            return self._session_id

    def send_frame(self, telemetry: dict[str, Any]) -> bool:
        """
        Send a single frame's exercise telemetry to the backend.

        Args:
            telemetry: Dict containing exercise type, angles, rep count, etc.

        Returns:
            True if successfully sent.
        """
        if not self._session_id:
            self.create_session()

        payload = {
            "session_id": self._session_id,
            "timestamp": time.time(),
            **telemetry,
        }

        try:
            resp = self._http.post(
                f"{self.base_url}/api/v1/exercises/frame",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.debug("Frame send failed (non-critical): %s", e)
            return False

    def get_coaching(self) -> Optional[dict]:
        """
        Request AI coaching feedback for the current session.

        Returns:
            Coaching feedback dict or None if unavailable.
        """
        if not self._session_id:
            return None

        try:
            resp = self._http.post(
                f"{self.base_url}/api/v1/sessions/{self._session_id}/coach",
                timeout=15.0,  # Longer timeout for AI generation
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.warning("Coaching request failed: %s", e)
            return None

    def end_session(self) -> None:
        """Mark the current session as complete."""
        if not self._session_id:
            return

        try:
            self._http.patch(
                f"{self.base_url}/api/v1/sessions/{self._session_id}",
                json={"ended_at": time.time()},
                timeout=self.timeout,
            )
            logger.info("Session ended: %s", self._session_id)
        except requests.RequestException as e:
            logger.warning("Failed to end session: %s", e)

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    def close(self) -> None:
        self._http.close()
