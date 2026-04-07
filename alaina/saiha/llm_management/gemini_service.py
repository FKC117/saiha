import os
import json
import logging
from typing import Any, Dict, List, Optional, Union
from google import genai
from google.genai import types

# Dedicated AI Audit Logger (Configured in settings.py)
ai_logger = logging.getLogger('saiha.ai')

class GeminiService:
    """
    Centralized service for Gemini 1.5 Pro/Flash integration.
    Uses the modern 'google-genai' (v1.0+) SDK.
    Hardened with Automated Audit Trails & Token Tracking.
    """
    def __init__(self, model_id: Optional[str] = None):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            ai_logger.error("GOOGLE_API_KEY not found in environment.")
            raise ValueError("Cloud analysis requires a valid Google API Key.")
        
        self.client = genai.Client(api_key=self.api_key)
        # Use provided model_id or fetch from .env, defaulting to gemini-2.0-flash
        self.model_id = model_id or os.getenv("DEFAULT_MODEL", "gemini-2.0-flash")

    def _log_interaction(self, prompt: str, response_text: str, usage: Any, session_id: Optional[str] = None):
        """
        Internal audit mechanism.
        Logs to file (ai.log) and DB (AIAuditLog).
        """
        tokens_in = usage.prompt_token_count if usage else 0
        tokens_out = usage.candidates_token_count if usage else 0
        
        # 1. File Logging (Structured Audit Trail)
        ai_logger.info(
            f"\n--- PROMPT ---\n{prompt}\n"
            f"--- RESPONSE ---\n{response_text}\n"
            f"--- METRICS --- [Model: {self.model_id}] [Tokens: {tokens_in} in / {tokens_out} out]"
        )
        
        # 2. Database Logging (Persistent Persistence)
        # Avoid circular imports by importing inside method
        try:
            from ..models import AIAuditLog, AnalysisSession
            session = None
            if session_id:
                session = AnalysisSession.objects.filter(id=session_id).first()
            
            AIAuditLog.objects.create(
                session=session,
                prompt=prompt,
                response=response_text,
                tokens_input=tokens_in,
                tokens_output=tokens_out,
                model_id=self.model_id
            )
        except Exception as e:
            ai_logger.error(f"Failed to save AIAuditLog to DB: {e}")

    def generate_content(self, prompt: str, system_instruction: Optional[str] = None, session_id: Optional[str] = None) -> str:
        """Standard single-turn generation with Audit Trail."""
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction
                )
            )
            self._log_interaction(prompt, response.text, response.usage_metadata, session_id)
            return response.text
        except Exception as e:
            ai_logger.error(f"Gemini generation failed: {e}")
            raise

    def get_intent_json(self, prompt: str, system_instruction: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Hardened Intent Extraction with Audit Trail.
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
            # Log raw text before parsing
            self._log_interaction(prompt, response.text, response.usage_metadata, session_id)
            return json.loads(response.text)
        except Exception as e:
            ai_logger.error(f"Gemini Intent extraction failed: {e}")
            raise  # Let AnalysisPlanner.create_plan() handle it and show user an error

# Global accessor
gemini_service = GeminiService()
