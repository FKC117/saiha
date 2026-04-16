import hashlib
import logging
from datetime import timedelta
from typing import Any, Dict, List, Optional, Union
from google import genai
from google.genai import types
from django.conf import settings
from .gemini_service import gemini_service

logger = logging.getLogger(__name__)

class ContextCacheManager:
    """
    Manages Gemini's Explicit Context Caching for large metadata.
    Avoids caching small chat history; focuses on static dataset schemas.
    Uses the modern 'google-genai' caches API.
    """
    def __init__(self, client: genai.Client):
        self.client = client

    def create_cache_if_large(self, content: str, model_id: str, ttl_minutes: int = 60) -> Optional[str]:
        """
        Creates a Gemini cache if the content exceeds a specific threshold.
        Returns the cache name/ID or None.
        Threshold: 32k characters (Approx 8k-10k tokens).
        """
        if len(content) < 32000:
            return None

        try:
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            display_name = f"schema_cache_{content_hash[:10]}"
            
            # Create a cache object via the GenAI SDK
            cache = self.client.caches.create(
                model=model_id,
                config=types.CreateCachedContentConfig(
                    display_name=display_name,
                    contents=[types.Content(parts=[types.Part(text=content)])],
                    # system_instruction="You are an expert data analyst using the following schema.",
                    ttl=f"{ttl_minutes}m", # timedelta string
                )
            )
            logger.info(f"Created Gemini Context Cache: {cache.name} for {display_name}")
            return cache.name
        except Exception as e:
            # We don't want to crash the analysis if caching fails
            logger.error(f"Failed to create Gemini Context Cache: {e}")
            return None

    def delete_cache(self, cache_name: str):
        """Standard cleanup."""
        try:
            self.client.caches.delete(name=cache_name)
            logger.info(f"Deleted Gemini Context Cache: {cache_name}")
        except Exception as e:
            logger.warning(f"Failed to delete Gemini Context Cache {cache_name}: {e}")

# Global accessor
context_cache_manager = ContextCacheManager(gemini_service.client)
