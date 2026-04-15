import json
from channels.generic.websocket import AsyncWebsocketConsumer

class NotificationConsumer(AsyncWebsocketConsumer):
    """
    Consumer for real-time task notifications.
    Allows the backend to push 'Analysis Started', 'Success', or 'Error' updates
    directly to the user's browser.
    """
    async def connect(self):
        # 1. Authentication guard — reject unauthenticated connections immediately
        user = self.scope.get('user')
        if not user or not user.is_authenticated:
            await self.close(code=4001)
            return
        self.user = user

        # 2. Extract session_id from the URL path
        self.session_id = self.scope['url_route']['kwargs'].get('session_id')

        if self.session_id:
            # 3. Ownership guard — verify this session belongs to the connecting user
            from saiha.models import AnalysisSession
            from channels.db import database_sync_to_async

            session_exists = await database_sync_to_async(
                AnalysisSession.objects.filter(id=self.session_id, user=user).exists
            )()

            if not session_exists:
                await self.close(code=4003)  # 4003 = Forbidden
                return

            self.group_name = f"notification_{self.session_id}"
        else:
            # Fallback: scope to per-user channel (no cross-user leakage)
            self.group_name = f"notification_user_{user.id}"

        # Join the identified group
        await self.channel_layer.group_add(
            self.group_name,
            self.channel_name
        )
        await self.accept()


    async def disconnect(self, close_code):
        if hasattr(self, 'group_name'):
            # Leave group
            await self.channel_layer.group_discard(
                self.group_name,
                self.channel_name
            )

    async def send_notification(self, event):
        """
        Handler for 'send_notification' messages.
        Ensures transparent pass-through of all fields (narratives, metadata, charts).
        """
        # Create a copy to avoid mutating the original event
        payload = dict(event)
        
        # --- Type Collision Fix (Bug 13) ---
        # Channels uses 'type' for the method name (send_notification).
        # But our AI agents use 'event_type' for the actual UI message category.
        # We map 'event_type' back to 'type' so the frontend JS routing works correctly.
        if 'event_type' in payload:
            payload['type'] = payload['event_type']
        else:
            payload['type'] = 'notification' # Default fallback
            
        # Send message to WebSocket
        await self.send(text_data=json.dumps(payload))
