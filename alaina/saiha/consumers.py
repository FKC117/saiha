import json
from channels.generic.websocket import AsyncWebsocketConsumer

class NotificationConsumer(AsyncWebsocketConsumer):
    """
    Consumer for real-time task notifications.
    Allows the backend to push 'Analysis Started', 'Success', or 'Error' updates
    directly to the user's browser.
    """
    async def connect(self):
        # We use a global group for now, or could use session-specific groups
        self.group_name = "analysis_notifications"
        
        # Join group
        await self.channel_layer.group_add(
            self.group_name,
            self.channel_name
        )
        await self.accept()

    async def disconnect(self, close_code):
        # Leave group
        await self.channel_layer.group_discard(
            self.group_name,
            self.channel_name
        )

    async def send_notification(self, event):
        """
        Handler for 'send_notification' messages.
        """
        message = event['message']
        # Send message to WebSocket
        await self.send(text_data=json.dumps({
            'type': 'notification',
            'message': message,
            'status': event.get('status', 'info'),
            'task_id': event.get('task_id'),
            'session_id': event.get('session_id')
        }))
