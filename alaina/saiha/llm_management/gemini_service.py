import os
import json
import logging
import hashlib
from datetime import timedelta
from django.utils import timezone
from typing import Any, Dict, List, Optional, Union
from google import genai
from google.genai import types, errors

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

    def _log_interaction(self, prompt: str, response_text: str, usage: Any, session_id: Optional[str] = None, user: Optional[Any] = None, metadata_status: str = "none"):
        """
        Internal audit mechanism.
        Logs to file (ai.log) and DB (AIAuditLog).
        Also updates UserQuota.

        File logging: metadata-only by default. Set AI_LOG_RAW_PAYLOADS=True in .env
        for raw payload capture (local debugging only — never enable in production).
        """
        from django.conf import settings as django_settings

        tokens_in = usage.prompt_token_count if (usage and usage.prompt_token_count) else 0
        tokens_out = usage.candidates_token_count if (usage and usage.candidates_token_count) else 0
        tokens_cached = usage.cached_content_token_count if (usage and usage.cached_content_token_count) else 0
        tokens_total = tokens_in + tokens_out # Note: cached tokens are usually billed differently, but for quota we count total impact

        # 1. File Logging — metadata only (safe for production log aggregators)
        ai_logger.info(
            "[AI Call] model=%s tokens_in=%d tokens_out=%d tokens_cached=%d session=%s user=%s prompt_len=%d",
            self.model_id,
            tokens_in,
            tokens_out,
            tokens_cached,
            session_id or "none",
            user.email if user and hasattr(user, 'email') else "anon",
            len(prompt),
        )
        # Optional full capture — disabled by default, enable only for local debugging
        if getattr(django_settings, 'AI_LOG_RAW_PAYLOADS', False):
            ai_logger.debug(
                "\n--- PROMPT ---\n%s\n--- RESPONSE ---\n%s",
                prompt,
                response_text,
            )

        # 2. Database Logging (AIAuditLog + UserQuota update)
        try:
            from ..models import AIAuditLog, AnalysisSession, UserQuota
            max_chars = getattr(django_settings, 'AI_AUDIT_LOG_MAX_CHARS', 2000)
            session = None
            final_user = user
            summary_len = 0
            wm_snap = {}

            if session_id:
                session = AnalysisSession.objects.filter(id=session_id).first()
                if session:
                    if not final_user:
                        final_user = session.user
                    summary_len = len(session.memory_summary)
                    wm_snap = session.working_memory

            # Create Audit Log
            AIAuditLog.objects.create(
                user=final_user,
                session=session,
                prompt=prompt[:max_chars],
                response=response_text[:max_chars],
                tokens_input=tokens_in,
                tokens_output=tokens_out,
                tokens_cached=tokens_cached,
                cache_hit=tokens_cached > 0,
                summary_length=summary_len,
                working_memory_snapshot=wm_snap,
                metadata_status=metadata_status,
                model_id=self.model_id
            )

            # Update Quota
            if final_user:
                quota, _ = UserQuota.objects.get_or_create(user=final_user)
                quota.current_tokens_used += tokens_total
                quota.save()

        except Exception as e:
            ai_logger.error("Failed to save AIAuditLog or update Quota: %s", e)

    def get_or_create_cache(self, session, static_context_str: str, system_instruction: str = None) -> Optional[str]:
        """
        Implements Deterministic Context Caching (v3). 
        Only creates a cache if it doesn't exist or has changed.
        Minimum threshold: 2,048 tokens (approx 8,000 characters).
        """
        # 1. Deterministic Hash Check
        # Versioning tool descriptions ensures cache reuse doesn't become logically stale
        TOOL_DESCRIPTIONS_VERSION = "v2-rolling-memory" 
        SYSTEM_PROMPT_VERSION = "v3.2-production"

        # Calculate schema hash to invalidate on structural changes (e.g. column rename)
        schema_hash = "none"
        if session and session.dataset:
            schema_data = session.dataset.get_column_types()
            schema_hash = hashlib.sha256(json.dumps(schema_data, sort_keys=True).encode()).hexdigest()

        hash_input = {
            "system_instruction": system_instruction,
            "static_context": static_context_str,
            "dataset_id": str(session.dataset.id) if session.dataset else "none",
            "model_id": self.model_id,
            "tool_version": TOOL_DESCRIPTIONS_VERSION,
            "prompt_version": SYSTEM_PROMPT_VERSION,
            "schema_hash": schema_hash
        }
        new_hash = hashlib.sha256(json.dumps(hash_input, sort_keys=True).encode()).hexdigest()

        # 2. Check for cache reuse
        if (session.llm_cache_id and 
            session.llm_cache_hash == new_hash and 
            session.llm_cache_expiry and 
            session.llm_cache_expiry > timezone.now()):
            ai_logger.info(f"Reusing Gemini Cache: {session.llm_cache_id} for session {session.id}")
            return session.llm_cache_id

        # 3. Threshold Guard: Only cache if large enough (Gemini limit: 4096 tokens ≈ 16k-20k chars)
        # Using 25,000 chars as a safe floor to avoid 400 INVALID_ARGUMENT.
        if len(static_context_str) < 25000:
            return None

        # 4. Create New Cache
        try:
            ai_logger.info(f"Creating new Gemini Context Cache for session {session.id}...")
            cache = self.client.caches.create(
                model=self.model_id,
                config=types.CreateCachedContentConfig(
                    display_name=f"session_{str(session.id)[:8]}",
                    system_instruction=system_instruction,
                    contents=[types.Content(role="user", parts=[types.Part(text=static_context_str)])],
                    ttl="3600s", # 1 hour
                )
            )
            
            # Save metadata
            session.llm_cache_id = cache.name
            session.llm_cache_hash = new_hash
            session.llm_cache_expiry = timezone.now() + timedelta(minutes=60)
            session.save(update_fields=['llm_cache_id', 'llm_cache_hash', 'llm_cache_expiry'])
            
            ai_logger.info(f"Created Gemini Cache: {cache.name}")
            return cache.name
        except Exception as e:
            ai_logger.error(f"Failed to create Gemini Cache: {e}")
            return None

    def generate_content(self, prompt: str, system_instruction: Optional[str] = None, session_id: Optional[str] = None, user: Optional[Any] = None, cache_name: Optional[str] = None, metadata_status: str = "none") -> str:
        """Standard single-turn generation with Audit Trail and Caching support."""
        try:
            config = types.GenerateContentConfig(
                system_instruction=system_instruction if not cache_name else None,
                cached_content=cache_name
            )
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=config
            )
            self._log_interaction(prompt, response.text, response.usage_metadata, session_id, user, metadata_status)
            return response.text
        except errors.ClientError as e:
            if e.code == 403 and cache_name:
                ai_logger.warning("Cache 403 Forbidden; invalidating and retrying without cache.")
                self._invalidate_session_cache(session_id)
                return self.generate_content(prompt, system_instruction, session_id, user, cache_name=None)
            raise
        except Exception as e:
            ai_logger.error(f"Gemini generation failed: {e}")
            raise

    def _invalidate_session_cache(self, session_id: str):
        """Standard cleanup for failed/expired caches."""
        if not session_id: return
        from ..models import AnalysisSession
        session = AnalysisSession.objects.filter(id=session_id).first()
        if session:
            session.llm_cache_id = None
            session.llm_cache_hash = None
            session.save(update_fields=['llm_cache_id', 'llm_cache_hash'])

    def get_intent_json(self, prompt: str, system_instruction: str, session_id: Optional[str] = None, user: Optional[Any] = None, cache_name: Optional[str] = None, metadata_status: str = "none") -> Dict[str, Any]:
        """
        Hardened Intent Extraction with Audit Trail and Cache support.
        """
        try:
            config = types.GenerateContentConfig(
                system_instruction=system_instruction if not cache_name else None,
                cached_content=cache_name,
                response_mime_type="application/json"
            )
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=config
            )
            # Log raw text before parsing
            self._log_interaction(prompt, response.text, response.usage_metadata, session_id, user, metadata_status)
            return json.loads(response.text)
        except errors.ClientError as e:
            if e.code == 403 and cache_name:
                ai_logger.warning("Cache 403 Forbidden; invalidating and retrying without cache.")
                self._invalidate_session_cache(session_id)
                return self.get_intent_json(prompt, system_instruction, session_id, user, cache_name=None)
            raise
        except Exception as e:
            ai_logger.error(f"Gemini Intent extraction failed: {e}")
            raise

# Global accessor
gemini_service = GeminiService()
