import uuid
import re
import logging
from django.utils import timezone
from django.db import transaction
from ..models import AnalysisSession, ChatMessage, AnalysisResult, Dataset

logger = logging.getLogger(__name__)

class SessionManager:
    """
    Manages the lifecycle of Analysis Sessions, ensuring each is tied to a Dataset.
    Ported and upgraded from the Quantly legacy project.
    """

    @staticmethod
    def get_or_create_session(user, dataset_id, session_name=None):
        """
        Retrieves an active session for the given dataset or creates a new one.
        """
        try:
            dataset = Dataset.objects.get(id=dataset_id, user=user)
            
            # Check for existing active session for this specific dataset
            session = AnalysisSession.objects.filter(
                user=user,
                dataset=dataset,
                is_active=True
            ).first()

            if session:
                session.last_activity = timezone.now()
                session.save(update_fields=['last_activity'])
                return session

            # Create new session if none exists
            if not session_name:
                session_name = f"Exploration: {dataset.name}"

            with transaction.atomic():
                session = AnalysisSession.objects.create(
                    user=user,
                    dataset=dataset,
                    session_name=session_name
                )
                
                # Add initial welcome message
                SessionManager.add_system_message(
                    session,
                    f"Session started for **{dataset.name}**. I'm ready to help you analyze this data. What would you like to know?"
                )

            return session

        except Dataset.DoesNotExist:
            logger.error(f"Dataset {dataset_id} not found for user {user.username}")
            return None
        except Exception as e:
            logger.error(f"Error in get_or_create_session: {str(e)}")
            return None

    @staticmethod
    def add_message(session, message_type, content, metadata=None):
        """
        Adds a message to the session and ensures it's tagged with the dataset context.
        """
        try:
            # Metadata upgrade: Ensure dataset context is always present
            meta = dict(metadata or {})
            if session.dataset:
                meta.setdefault('dataset_id', str(session.dataset.id))

            # Logic upgrade: Strip dangerous or stray template tags from AI content
            if message_type == 'ai':
                content = SessionManager._sanitize_content(content)

            message = ChatMessage.objects.create(
                session=session,
                message_type=message_type,
                content=content,
                metadata=meta
            )

            # Update session activity
            session.chat_message_count += 1
            session.save(update_fields=['chat_message_count', 'last_activity'])
            
            return message
        except Exception as e:
            logger.error(f"Error adding message: {str(e)}")
            return None

    @staticmethod
    def add_system_message(session, content):
        return SessionManager.add_message(session, 'system', content)

    @staticmethod
    def add_user_message(session, content):
        return SessionManager.add_message(session, 'user', content)

    @staticmethod
    def add_ai_message(session, content, metadata=None):
        return SessionManager.add_message(session, 'ai', content, metadata)

    @staticmethod
    def add_analysis_result(session, tool_used, result_data, interpretation=None):
        """
        Saves a detailed analysis result and logs it in the chat history.
        """
        try:
            with transaction.atomic():
                # 1. Create the persistent result record
                result = AnalysisResult.objects.create(
                    session=session,
                    tool_used=tool_used,
                    result_data=result_data,
                    ai_interpretation=interpretation
                )

                # 2. Add an analysis_result type message to the chat history
                SessionManager.add_message(
                    session,
                    'analysis_result',
                    f"Successfully executed **{tool_used}** analysis.",
                    {
                        'tool': tool_used,
                        'result_id': str(result.id),
                        'has_interpretation': bool(interpretation)
                    }
                )

                # 3. Increment analysis counter
                session.analysis_count += 1
                session.save(update_fields=['analysis_count', 'last_activity'])

                return result
        except Exception as e:
            logger.error(f"Error adding analysis result: {str(e)}")
            return None

    @staticmethod
    def _sanitize_content(content):
        """
        Upgraded sanitization to prevent stray markup or template tags from appearing.
        """
        if not isinstance(content, str):
            return content
            
        # Remove Django-style template comments that models sometimes hallucinate
        content = re.sub(r"\{#([\s\S]*?)#\}", "", content)
        return content.strip()

    @staticmethod
    def get_history(session, limit=100):
        """
        Returns the chat messages for the session in chronological order.
        """
        return ChatMessage.objects.filter(session=session).order_by('created_at')[:limit]
