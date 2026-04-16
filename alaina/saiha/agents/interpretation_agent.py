import logging
import json
from typing import Dict, Any, Optional, List
from ..llm_management.gemini_service import gemini_service
from ..models import AnalysisResult, AnalysisSession
from .context_builder import ContextBuilder

logger = logging.getLogger(__name__)

class InterpretationAgent:
    """
    The 'Interpreter' Role.
    Takes structured tool outputs → generates natural language insights.
    Runs AFTER the Celery Task (Executor) has finished.
    """
    _SYSTEM_INSTRUCTION_STATIC = """
    You are a Senior Consulting Data Analyst for the ChatFlow system.
    Your role is to turn raw statistical tool outputs into high-fidelity business narratives.
    
    OUTPUT STRUCTURE (MANDATORY):
    1. **Key Takeaway**: A single, powerful sentence summarizing the main finding.
    2. **Deep Dive**: 3-4 bullet points explaining the statistical significance and trends.
    3. **Anomalies**: Mention any outliers, data quality issues, or unexpected skews.
    4. **Next Step**: A proactive recommendation for the NEXT DIFFERENT logical analysis. 
       - **PARAM-AWARE SILENCE**: Do NOT suggest running the same tool again with the same parameters.
       - Instead, suggest a DIFFERENT variable (e.g. 'Now try analyzing Income outliers') or a DIFFERENT analysis (e.g. 'Now try a correlation matrix').
       - If the query is fully satisfied, say: "Analysis complete. What would you like to explore next?"
    
    TONE: Professional, insightful, and authoritative. Avoid jargon unless explaining it.
    
    METADATA EXTRACTION (STRICT):
    At the very end of your response, after any Next Steps, you MUST append a metadata block exactly like this:
    [METADATA]
    type: <one_word_category_of_finding>
    target: <primary_column_of_interest_or_N/A>
    columns: <col1>|<col2>|<col3>
    
    Example:
    [METADATA]
    type: positive_correlation
    target: Age
    columns: Age|Income
    """

    def __init__(self, model_id: Optional[str] = None):
        # Use provided model_id or default to the service's model
        self.model_id = model_id or gemini_service.model_id

    def interpret_result(self, result_id: str, final_artifacts: List[Dict[str, Any]] = None) -> str:
        """
        Interprets a specific analysis result.
        """
        try:
            result = AnalysisResult.objects.get(id=result_id)
            if result.status != AnalysisResult.Status.SUCCESS:
                return f"Analysis (Status: {result.status}) did not complete successfully; skipped interpretation."

            tool_used = result.tool_used
            data_json = result.result_data or {}
            query = result.query
            session = result.session

            # Use provided artifacts (preferred) or fallback to data_json['artifacts']
            artifacts = final_artifacts if final_artifacts is not None else data_json.get('artifacts', [])

            # 1. Fetch Schema for Cache Consistency
            schema_text = ""
            if session.dataset:
                columns_meta = session.dataset.columns.all()
                schema_text = "\n".join([f"- {c.column_name} ({c.data_type})" for c in columns_meta])

            # 2. Build Static Context (FOR CACHING)
            # Use empty tool list for interpreter if not needed, but keeps hash same if rules/schema same
            static_context = ContextBuilder.build_static_context(
                self._SYSTEM_INSTRUCTION_STATIC,
                schema_text,
                tools_json="{}" 
            )

            # 3. Get or Create Gemini Context Cache
            cache_id = gemini_service.get_or_create_cache(
                session=session,
                static_context_str=static_context,
                system_instruction=self._SYSTEM_INSTRUCTION_STATIC
            )

            # 4. Prepare Tool Output (Slimmed)
            import copy
            slim_data = copy.deepcopy(data_json)
            if 'artifacts' in slim_data:
                for art in slim_data['artifacts']:
                    if art.get('type') in ['plot', 'image', 'chart'] and 'content' in art:
                        art['content'] = "[Base64 Image Data Stripped for LLM Interpretation]"
            
            tool_results_text = f"The user asked '{query}' using the '{tool_used}' tool.\nTOOL OUTPUT JSON: {json.dumps(slim_data)}"

            # 5. Build Dynamic Prompt (Rolling Memory + Results)
            dynamic_prompt = ContextBuilder.build_interpreter_context(session, tool_results_text)

            # 6. Call Gemini
            interpretation = gemini_service.generate_content(
                prompt=dynamic_prompt,
                system_instruction=self._SYSTEM_INSTRUCTION_STATIC, 
                session_id=str(session.id), 
                user=session.user,
                cache_name=cache_id,
                metadata_status="none" 
            )

            # Extract metadata, but never let metadata/summarization issues block
            # delivery of the finished answer to the UI.
            from .memory_manager import MemoryManager
            try:
                clean_interpretation, metadata = MemoryManager.extract_metadata_footer(interpretation, session=session)
            except Exception as metadata_error:
                logger.error(
                    "Metadata extraction failed for result %s in session %s: %s",
                    result_id,
                    session.id,
                    metadata_error,
                    exc_info=True,
                )
                clean_interpretation = interpretation
                metadata = {}

            if not metadata:
                logger.warning(f"Metadata extraction failed for session {session.id}. Applying last_valid_metadata fallback.")
                metadata = session.last_valid_metadata or {}

            if 'last_tool' not in metadata:
                metadata['last_tool'] = tool_used

            # Persist the final interpretation first so the UI can render it even
            # if later memory bookkeeping fails.
            result.ai_interpretation = clean_interpretation
            result.save(update_fields=['ai_interpretation'])

            # --- HARDENED METADATA FOR WEB SOCKETS (Elite v3.8) ---
            # Enforce that 'artifacts' is the primary channel for UI rendering.
            message_metadata = {
                "id": str(result.id),
                "tool_used": tool_used,
                "data": data_json.get('data', {}),
                "artifacts": artifacts, # PRE-NORMALIZED LIST
                "query": query
            }

            # --- Persistence Layer (Bug Fix: Bug 12.4.2) ---
            # Record the AI response in the persistent chat history.
            from ..session_management.session_manager import SessionManager
            SessionManager.add_ai_message(
                session, 
                clean_interpretation, 
                metadata=message_metadata
            )
            
            # --- BROADCAST FINAL RESULT VIA WEBSOCKET ---
            from asgiref.sync import async_to_sync
            from channels.layers import get_channel_layer
            
            channel_layer = get_channel_layer()
            if channel_layer:
                async_to_sync(channel_layer.group_send)(
                    f"notification_{str(session.id)}",
                    {
                        "type": "send_notification", 
                        "event_type": "agent_message",
                        "message_type": "ai",
                        "content": clean_interpretation,
                        "metadata": message_metadata,
                        "status": "success",
                        "session_id": str(session.id)
                    }
                )

            # Memory bookkeeping is helpful but non-critical. It should never
            # prevent the answer from reaching the user.
            try:
                MemoryManager.decay_stale_state(session)
                MemoryManager.update_working_memory(session, metadata=metadata)
                MemoryManager.update_summary(session, gemini_service)
            except Exception as memory_error:
                logger.error(
                    "Post-interpretation memory update failed for result %s in session %s: %s",
                    result_id,
                    session.id,
                    memory_error,
                    exc_info=True,
                )

            logger.info(f"Generated, broadcast, and summarized interpretation for Result {result_id}")
            return clean_interpretation

        except Exception as e:
            logger.error(f"Interpretation failed for Result {result_id}: {e}", exc_info=True)
            return f"Interpretation error: {str(e)}"

# Global accessor
interpretation_agent = InterpretationAgent()
