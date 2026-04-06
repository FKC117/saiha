import logging
from typing import Dict, Any, Optional
from ..llm_management.gemini_service import gemini_service
from ..models import AnalysisResult

logger = logging.getLogger(__name__)

class InterpretationAgent:
    """
    The 'Interpreter' Role.
    Takes structured tool outputs → generates natural language insights.
    Runs AFTER the Celery Task (Executor) has finished.
    """
    def __init__(self, model_id: Optional[str] = None):
        # Use provided model_id or default to the service's model
        self.model_id = model_id or gemini_service.model_id

    def interpret_result(self, result_id: str) -> str:
        """
        Interprets a specific analysis result.
        """
        try:
            result = AnalysisResult.objects.get(id=result_id)
            if result.status != "SUCCESS":
                return "Analysis did not complete successfully; skipped interpretation."

            tool_used = result.tool_used
            data_json = result.result_data
            query = result.query
            session = result.session

            # Build Interpretation Prompt
            system_instruction = f"""
            You are a Senior Consulting Data Analyst for the ChatFlow system.
            Your role is to turn raw statistical tool outputs into high-fidelity business narratives.
            
            CONTEXT: The user asked '{query}' using the '{tool_used}' tool.
            
            OUTPUT STRUCTURE (MANDATORY):
            1. **Key Takeaway**: A single, powerful sentence summarizing the main finding.
            2. **Deep Dive**: 3-4 bullet points explaining the statistical significance and trends.
            3. **Anomalies**: Mention any outliers, data quality issues, or unexpected skews.
            4. **Next Step**: A proactive recommendation for the next logical analysis (e.g., 'Check for correlation between X and Y').
            
            TONE: Professional, insightful, and authoritative. Avoid jargon unless explaining it.
            """

            prompt = f"TOOL OUTPUT JSON: {data_json}\nGenerate the report."

            interpretation = gemini_service.generate_content(prompt, system_instruction)
            
            # Update the result record
            result.ai_interpretation = interpretation
            result.save()
            
            # --- BROADCAST FINAL RESULT VIA WEBSOCKET ---
            from asgiref.sync import async_to_sync
            from channels.layers import get_channel_layer
            
            channel_layer = get_channel_layer()
            if channel_layer:
                async_to_sync(channel_layer.group_send)(
                    f"notification_{str(session.id)}",
                    {
                        "type": "send_notification", # Consumer maps this to 'notification' type but we add event_type
                        "event_type": "agent_message",
                        "message_type": "ai",
                        "content": interpretation,
                        "metadata": {
                            "id": str(result.id),
                            "tool_used": tool_used,
                            "data": data_json.get('data', {}),
                            "artifacts": data_json.get('artifacts', []),
                            "query": query
                        },
                        "status": "success",
                        "session_id": str(session.id)
                    }
                )
            
            logger.info(f"Generated and broadcast interpretation for Result {result_id}")
            return interpretation

        except Exception as e:
            logger.error(f"Interpretation failed for Result {result_id}: {e}")
            return f"Interpretation error: {str(e)}"

# Global accessor
interpretation_agent = InterpretationAgent()
