"""
ASGI config for alaina project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/asgi/
"""

import os
from django.core.asgi import get_asgi_application

# 1. Set environment first
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "alaina.settings")

# 2. Initialize Django ASGI application
django_asgi_app = get_asgi_application()

# 3. Import Channels components after Django is ready
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
import saiha.routing

application = ProtocolTypeRouter({
    "http": django_asgi_app,
    "websocket": AuthMiddlewareStack(
        URLRouter(
            saiha.routing.websocket_urlpatterns
        )
    ),
})
