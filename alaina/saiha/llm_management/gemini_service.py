import os
import logging
from typing import Any, Dict, List, Optional, Union
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

class GeminiService:
    """
    Centralized service for Gemini 1.5 Pro/Flash integration.
    Uses the modern 'google-genai' (v1.0+) SDK.
    Handles chat, intent detection, and results interpretation.
    """
    def __init__(self, model_id: str = "gemini-1.5-flash"):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            logger.error("GOOGLE_API_KEY not found in environment.")
            raise ValueError("Cloud analysis requires a valid Google API Key.")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model_id = model_id

    def generate_content(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """Standard single-turn generation."""
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise

    def get_intent_json(self, prompt: str, system_instruction: str) -> Dict[str, Any]:
        """
        Hardened Intent Extraction.
        Enforces JSON output via Response MIME Type.
        """
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json"
                )
            )
            import json
            return json.loads(response.text)
        except Exception as e:
            logger.error(f"Gemini Intent extraction failed: {e}")
            return {"error": str(e)}

# Global accessor
gemini_service = GeminiService()
