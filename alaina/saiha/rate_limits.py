import time
from functools import wraps

from django.conf import settings
from django.core.cache import cache
from django.http import JsonResponse


_UNIT_SECONDS = {
    's': 1,
    'm': 60,
    'h': 3600,
    'd': 86400,
}


def parse_rate(rate_value: str) -> tuple[int, int]:
    try:
        amount_str, unit = rate_value.split('/', 1)
        amount = int(amount_str)
        seconds = _UNIT_SECONDS[unit]
    except (ValueError, KeyError) as exc:
        raise ValueError(f"Invalid rate limit value: {rate_value}") from exc

    return amount, seconds


def get_rate_for_scope(scope: str, default_rate: str) -> tuple[int, int]:
    configured = getattr(settings, 'PHASE2_RATE_LIMITS', {}).get(scope, default_rate)
    return parse_rate(configured)


def build_identity(request):
    if getattr(request, 'user', None) and request.user.is_authenticated:
        return f"user:{request.user.pk}"
    return f"ip:{request.META.get('REMOTE_ADDR', 'unknown')}"


def rate_limit(scope: str, default_rate: str, methods=('POST',), message='Too many requests. Please slow down.'):
    def decorator(view_func):
        @wraps(view_func)
        def wrapped(request, *args, **kwargs):
            if request.method not in methods:
                return view_func(request, *args, **kwargs)

            allowed, window_seconds = get_rate_for_scope(scope, default_rate)
            identity = build_identity(request)
            bucket = int(time.time() // window_seconds)
            cache_key = f"ratelimit:{scope}:{identity}:{bucket}"
            count = cache.get(cache_key, 0) + 1
            cache.set(cache_key, count, timeout=window_seconds + 5)

            if count > allowed:
                retry_after = max(window_seconds - int(time.time() % window_seconds), 1)
                response = JsonResponse(
                    {
                        'status': 'error',
                        'message': message,
                        'scope': scope,
                    },
                    status=429,
                )
                response['Retry-After'] = str(retry_after)
                return response

            return view_func(request, *args, **kwargs)

        return wrapped

    return decorator
