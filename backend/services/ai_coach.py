"""
AI Coaching Service — Azure OpenAI Integration.

Generates personalized exercise form feedback using GPT-4o
via Azure AI Foundry. Analyzes session telemetry including
joint angles, rep counts, and range of motion data.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

from openai import AzureOpenAI

logger = logging.getLogger(__name__)


class AICoach:
    """
    AI-powered coaching feedback generator.

    Uses Azure OpenAI (GPT-4o) to analyze exercise session data
    and produce actionable form corrections and training advice.
    """

    SYSTEM_PROMPT = """You are FitForm AI Coach, an expert exercise science advisor.
You analyze biomechanical data from computer vision pose estimation to provide
actionable coaching feedback.

You receive session data including:
- Exercise types performed and rep counts
- Joint angle measurements (degrees) for key joints
- Range of motion (ROM) statistics showing min/max angles per joint

Provide feedback that is:
1. Specific and actionable (reference exact joints and angles)
2. Encouraging but honest about form issues
3. Based on established exercise science principles
4. Prioritized by injury risk (most important corrections first)

Format your response as JSON with these fields:
{
    "feedback": "2-3 paragraph natural language summary",
    "form_score": 7.5,  // 0-10 scale
    "recommendations": ["specific recommendation 1", "specific recommendation 2", ...]
}

Reference ranges for good form:
- Air Squat: Knee angle should reach 70-90° at bottom, hip crease below knee
- Push-Up: Elbow angle should reach 80-100° at bottom, full extension at top
- Sit-Up: Hip angle should reach 40-60° at top of movement"""

    def __init__(self) -> None:
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

        if not endpoint or not api_key:
            logger.warning(
                "Azure OpenAI credentials not configured. "
                "Coaching feedback will use mock responses."
            )
            self._client = None
            self._deployment = None
            return

        self._client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )
        self._deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        logger.info("AICoach initialized | deployment=%s", self._deployment)

    def generate_feedback(
        self,
        session_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Generate coaching feedback for an exercise session.

        Args:
            session_data: Dict with exercises, rep counts, angles, ROM stats.

        Returns:
            Dict with feedback, form_score, and recommendations.
        """
        if not self._client:
            return self._mock_feedback(session_data)

        user_message = (
            f"Analyze this exercise session and provide coaching feedback:\n\n"
            f"```json\n{json.dumps(session_data, indent=2)}\n```"
        )

        try:
            response = self._client.chat.completions.create(
                model=self._deployment,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.7,
                max_tokens=800,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            result = json.loads(content)
            result["model"] = self._deployment
            logger.info("Coaching feedback generated (score: %s)", result.get("form_score"))
            return result

        except Exception as e:
            logger.error("Azure OpenAI error: %s", e)
            return self._mock_feedback(session_data)

    @staticmethod
    def _mock_feedback(session_data: dict) -> dict[str, Any]:
        """Fallback mock feedback when Azure OpenAI is not available."""
        exercises = session_data.get("exercises", {})
        total_reps = sum(exercises.values())

        return {
            "feedback": (
                f"Session completed with {total_reps} total repetitions across "
                f"{len([e for e, c in exercises.items() if c > 0])} exercise types. "
                f"Connect Azure OpenAI for AI-powered form analysis and personalized coaching."
            ),
            "form_score": None,
            "recommendations": [
                "Configure AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY for AI coaching",
                "Ensure camera is positioned to capture full body in frame",
                "Maintain consistent distance from camera between sessions",
            ],
            "model": "mock",
        }
