import json
import logging
from .connection import redis_manager

logger = logging.getLogger(__name__)

class AnalysisCache:
    """
    Specialized caching for ephemeral Analysis payloads.
    Stores metadata that doesn't need full DB persistence but is too large for Celery.
    """
    def __init__(self, ttl=3600): # 1 Hour Default
        self.ttl = ttl

    def set_payload(self, task_id, data):
        """Stores large JSON payloads in Redis with an expiration."""
        try:
            key = f"analysis:payload:{task_id}"
            redis_manager.client.setex(key, self.ttl, json.dumps(data))
            return True
        except Exception as e:
            logger.error(f"Failed to cache analysis payload: {e}")
            return False

    def get_payload(self, task_id):
        """Retrieves and deserializes payloads."""
        try:
            key = f"analysis:payload:{task_id}"
            val = redis_manager.client.get(key)
            return json.loads(val) if val else None
        except Exception as e:
            logger.error(f"Failed to retrieve analysis payload: {e}")
            return None

# Global accessor
analysis_cache = AnalysisCache()
