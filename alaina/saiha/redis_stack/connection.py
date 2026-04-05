import redis
import logging
from django.conf import settings

logger = logging.getLogger(__name__)

class RedisConnectionManager:
    """
    Singleton manager for Redis Stack connections with circuit-breaker awareness.
    Guarantees no silent fallback to sync mode.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RedisConnectionManager, cls).__new__(cls)
            cls._instance._client = None
        return cls._instance

    @property
    def client(self):
        """Returns the Redis client or raises an error if unavailable."""
        if self._client is None:
            self._connect()
        return self._client

    def _connect(self):
        """Internal connection logic with failure protection."""
        try:
            url = getattr(settings, 'REDIS_URL', 'redis://localhost:6379/0')
            self._client = redis.Redis.from_url(
                url, 
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2
            )
            self._client.ping()
            logger.info("Successfully connected to Redis Stack.")
        except (redis.ConnectionError, redis.TimeoutError) as e:
            self._client = None
            logger.error(f"FATAL: Redis connection failed. No silent fallback allowed. Error: {e}")
            raise ConnectionError("Redis Analysis Broker is currently unavailable.")

    def is_alive(self):
        """Check if Redis is reachable without raising exceptions."""
        try:
            if self._client:
                return self._client.ping()
            self._connect()
            return True
        except Exception:
            return False

# Global accessor
redis_manager = RedisConnectionManager()
