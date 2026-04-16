import logging
from typing import List, Dict, Any, Optional
from django.conf import settings
from django.core.cache import cache
from ..models import AnalysisSession, AnalysisResult, ChatMessage, Dataset
from .analysis_planner import analysis_planner
from .param_corrector import ParamCorrector
from .interpretation_agent import interpretation_agent
from ..analysis_tools.registry import tool_registry
from ..celery_tasks.analysis_tasks import execute_analysis_task, send_ws_notification
from ..session_management.session_manager import SessionManager
from .memory_manager import MemoryManager

logger = logging.getLogger(__name__)

class AnalysisAgent:
    """
    The Centralized Orchestrator for the ChatFlow AI agent.
    Role: Gathers intent, corrects parameters, and dispatches to Celery.
    Prevents LLM from directly controlling tool execution.
    """
    def __init__(self, session_or_id):
        if isinstance(session_or_id, AnalysisSession):
            self.session = session_or_id
        else:
            self.session = AnalysisSession.objects.get(id=session_or_id)
        self.dataset = self.session.dataset

    def process_query(self, query: str) -> List[str]:
        """
        Processes a natural language query with context-awareness and real-time status.
        """
        can_dispatch, guard_message = self._guard_dispatch()
        if not can_dispatch:
            logger.warning(f"Dispatch blocked for session {self.session.id}: {guard_message}")
            send_ws_notification(
                guard_message,
                status="error",
                session_id=str(self.session.id)
            )
            return []

        # --- Persistence Layer ---
        # Record the user's intent in the persistent chat history.
        if not self.session.messages.filter(message_type='user', content=query).exists():
            SessionManager.add_user_message(self.session, query)

        # 0. Memory Check (Auto-summarize if count threshold reached)
        if MemoryManager.should_update_summary(self.session):
            from ..llm_management.gemini_service import gemini_service
            MemoryManager.update_summary(self.session, gemini_service)

        # Notify UI: Planning Stage
        send_ws_notification(
            "Planning your analysis strategy...",
            status="planning",
            session_id=str(self.session.id)
        )

        # 1. Dataset Health Check
        if not self.dataset:
            logger.warning(f"Analysis attempted on session {self.session.id} without a dataset.")
            send_ws_notification(
                "No dataset is active. Please upload or select a database first.",
                status="error",
                session_id=str(self.session.id)
            )
            return []

        # 2. Get Schema Metadata
        columns_meta = self.dataset.columns.all()
        schema_text = "\n".join([f"- {c.column_name} ({c.data_type})" for c in columns_meta])
        column_names = [c.column_name for c in columns_meta]

        # 2. Planning (LLM Intent Detection via ContextBuilder/Caching)
        intents = analysis_planner.create_plan(
            query=query, 
            schema_text=schema_text, 
            session_id=str(self.session.id), 
            user=self.session.user
        )
        if not intents:
            logger.warning(f"No intents generated for query: '{query}'")
            # --- Saturation Check (Elite v3.5) ---
            # If history exists, we assume the LLM found the query satisfied by Rule 1.
            if ChatMessage.objects.filter(session=self.session).exists():
                send_ws_notification(
                    "I've already analyzed that for you! Check the history above or let me know if you want to perform a different analysis.",
                    status="success",
                    session_id=str(self.session.id)
                )
            else:
                send_ws_notification(
                    "I'm here to help, but I couldn't determine a plan. Try rephrasing or asking for a specific column.",
                    status="error",
                    session_id=str(self.session.id)
                )
            return []

        # --- explicit Saturation Signal (Elite v3.5) ---
        if len(intents) == 1 and intents[0].get('tool') == 'chat' and intents[0].get('params', {}).get('message') == "ALREADY_DONE":
            send_ws_notification(
                "I've already analyzed that for you in our history. What else would you like me to look at?",
                status="success",
                session_id=str(self.session.id)
            )
            return []

        intents = self._apply_tool_cap(intents)
        if not intents:
            return []

        # 3. Hybrid Correction Layer
        corrector = ParamCorrector(column_names)
        task_ids = []

        # Notify UI about planned tasks
        send_ws_notification(
            f"Orchestrating {len(intents)} analytical step(s)...",
            status="dispatching",
            session_id=str(self.session.id)
        )

        for intent in intents:
            # Handle potential LLM key name hallucination ('tool' vs 'name')
            tool_name = intent.get('tool') or intent.get('name')
            raw_params = intent.get('params') or intent.get('parameters') or {}
            
            # --- SECURITY & REGISTRY WHITELIST (Elite v3.6) ---
            # Bypass registry for internal 'chat' tool used for planning notifications.
            if tool_name == 'chat':
                # For chat intents, we just skip execution and move to next intent
                # (Actual chat responses for saturation are handled before the loop)
                continue

            # We must flatten this into a dictionary: {"col": "Age"}
            if isinstance(raw_params, list):
                normalized_params = {}
                for p in raw_params:
                    if isinstance(p, dict):
                        # Handle both 'name'/'value' and 'key'/'value' patterns
                        p_name = p.get('name') or p.get('key') or p.get('parameter')
                        p_val = p.get('value')
                        if p_name:
                            normalized_params[p_name] = p_val
                raw_params = normalized_params
            
            # A. Whitelist Validation (Elite v3.7 Security Layer)
            if tool_name == 'chat':
                # Special internal tool for planning signals; skips execution.
                continue

            tool_instance = tool_registry.get_tool(tool_name)
            if not tool_instance:
                # Security: Block any unverified tools hallucininated by the LLM
                logger.warning(f"Tool '{tool_name}' blocked by registry whitelist.")
                send_ws_notification(
                    f"Security Error: Tool '{tool_name}' is not authorized for this session.",
                    status="error",
                    session_id=str(self.session.id)
                )
                continue

            # B. Schema-Aware Correction (uses tool's own ToolParameterSet contract)
            # Fix column names, enum values, type coercion, and fill defaults
            # before the tool is even initialized.
            try:
                schema = tool_instance.get_parameters_schema()
            except Exception as schema_err:
                logger.warning(f"Could not fetch schema for '{tool_name}': {schema_err}. Using legacy corrector.")
                schema = None
            
            try:
                corrected_params = corrector.apply_to_params(raw_params, schema)
            except ValueError as ve:
                # Required param missing with no default — fail loud, not silent
                logger.error(f"Param validation failed for '{tool_name}': {ve}")
                send_ws_notification(
                    f"Could not run '{tool_name}': {ve}",
                    status="error",
                    session_id=str(self.session.id)
                )
                continue
            
            # C. Create DB Record (Pending State) for Traceability
            # Using a deterministic dedup_id: session_id + query_hash + tool_idx
            import hashlib
            dedup_id = hashlib.sha256(f"{self.session.id}_{query}_{tool_name}".encode()).hexdigest()
            
            # 1. Create the Result Placeholder (Atoms of Traceability)
            result_record, created = AnalysisResult.objects.get_or_create(
                dedup_id=dedup_id,
                defaults={
                    "session": self.session,
                    "tool_used": tool_name,
                    "query": query,
                    "status": AnalysisResult.Status.PENDING
                }
            )

            # NOTE: We only pass reference IDs to Celery (Pass-by-Reference)
            # This follows the 'Hardened Payload' rule.
            try:
                # 2. Dispatch to Hardened Async Infrastructure
                # Using keyword arguments for version-agnostic stability
                task = execute_analysis_task.apply_async(
                    kwargs={
                        "result_id": str(result_record.id),
                        "session_id": str(self.session.id),
                        "tool_name": tool_name,
                        "params": corrected_params,
                        "dedup_id": dedup_id,
                        "query": query
                    }
                )
                task_ids.append(task.id)
                logger.info(f"Dispatched Hardened Task: {tool_name} (ID: {task.id})")
            except Exception as e:
                logger.error(f"Failed to dispatch task for {tool_name}: {e}")

        return task_ids

    def _guard_dispatch(self) -> tuple[bool, str | None]:
        cooldown_seconds = getattr(settings, 'ANALYSIS_SESSION_COOLDOWN_SECONDS', 5)
        cooldown_key = f"analysis-dispatch-cooldown:{self.session.id}"

        if cache.get(cooldown_key):
            return False, f"Please wait {cooldown_seconds} seconds before sending another analysis request."

        max_active = getattr(settings, 'ANALYSIS_MAX_ACTIVE_TASKS_PER_SESSION', 5)
        active_tasks = AnalysisResult.objects.filter(
            session=self.session,
            status__in=[AnalysisResult.Status.PENDING, AnalysisResult.Status.RUNNING],
        ).count()
        if active_tasks >= max_active:
            return False, "This session already has too many analyses in progress. Please wait for current tasks to finish."

        cache.set(cooldown_key, True, timeout=cooldown_seconds)
        return True, None

    def _apply_tool_cap(self, intents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        max_tools = getattr(settings, 'ANALYSIS_MAX_TOOLS_PER_REQUEST', 3)
        if len(intents) <= max_tools:
            return intents

        logger.warning(
            "Planner returned %s tools for session %s; truncating to %s.",
            len(intents),
            self.session.id,
            max_tools,
        )
        send_ws_notification(
            f"I planned {len(intents)} analyses, but I can only run {max_tools} at a time. Starting the first {max_tools}.",
            status="warning",
            session_id=str(self.session.id)
        )
        return intents[:max_tools]

# Accessor factory
def get_analysis_agent(session_or_id):
    return AnalysisAgent(session_or_id)
