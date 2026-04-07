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
            if result.status != AnalysisResult.Status.SUCCESS:
                return f"Analysis (Status: {result.status}) did not complete successfully; skipped interpretation."

            tool_used = result.tool_used
            data_json = result.result_data
            query = result.query
            session = result.session

            # Build Interpretation Prompt
            system_instruction = f"""
            You are a Senior Consulting Data Analyst for the ChatFlow system.
            Your role is to turn raw statistical tool outputs into high-fidelity business narratives.
            
            CONTEXT: The user asked '{query}' using the '{tool_used}' tool.
            HISTORY: Ensure you do NOT suggest a 'Next Step' that was already completed in the session history.
            
            OUTPUT STRUCTURE (MANDATORY):
            1. **Key Takeaway**: A single, powerful sentence summarizing the main finding.
            2. **Deep Dive**: 3-4 bullet points explaining the statistical significance and trends.
            3. **Anomalies**: Mention any outliers, data quality issues, or unexpected skews.
            4. **Next Step**: A proactive recommendation for the NEXT DIFFERENT logical analysis. 
               - **PARAM-AWARE SILENCE**: If the tool '{tool_used}' just ran effectively for the user's intent, do NOT suggest running it again with the same parameters.
               - Instead, suggest a DIFFERENT variable (e.g. 'Now try analyzing Income outliers') or a DIFFERENT analysis (e.g. 'Now try a correlation matrix').
               - If the query is fully satisfied, say: "Analysis complete. What would you like to explore next?"
            
            TONE: Professional, insightful, and authoritative. Avoid jargon unless explaining it.
            """

            # --- SLIM DATA FOR LLM (Bug 13.6: Prevent Token Overflow) ---
            # Strip heavy Base64 images from the prompt, but KEEP them in the final broadcast.
            import copy
            slim_data = copy.deepcopy(data_json)
            if 'artifacts' in slim_data:
                for art in slim_data['artifacts']:
                    if art.get('type') in ['plot', 'image', 'chart'] and 'content' in art:
                        # Keep the label and metadata, but remove the binary blob
                        art['content'] = "[Base64 Image Data Stripped for LLM Interpretation]"
            
            prompt = f"TOOL OUTPUT JSON: {slim_data}\nGenerate the report."

            interpretation = gemini_service.generate_content(prompt, system_instruction)
            
            # Update the result record
            result.ai_interpretation = interpretation
            result.save()

            # --- Persistence Layer (Bug Fix: Bug 12.4.2) ---
            # Record the AI response in the persistent chat history.
            # This ensures the narrative and its metadata (for charts) survives refresh.
            from ..session_management.session_manager import SessionManager
            SessionManager.add_ai_message(
                session, 
                interpretation, 
                metadata={
                    "id": str(result.id),
                    "tool_used": tool_used,
                    "data": data_json.get('data', {}),
                    "artifacts": data_json.get('artifacts', []),
                    "query": query
                }
            )
            
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
