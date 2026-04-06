from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    # Primary: Session-isolated route
    re_path(r'ws/notifications/(?P<session_id>[^/]+)/?$', consumers.NotificationConsumer.as_asgi()),
    # Fallback: Generic route for legacy tabs/general notifications
    re_path(r'ws/notifications/?$', consumers.NotificationConsumer.as_asgi()),
]
