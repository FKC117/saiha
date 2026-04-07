import logging
from typing import List, Dict, Any, Optional
from ..models import AnalysisSession, AnalysisResult, Dataset
from .analysis_planner import analysis_planner
from .param_corrector import ParamCorrector
from .interpretation_agent import interpretation_agent
from ..analysis_tools.registry import tool_registry
from ..celery_tasks.analysis_tasks import execute_analysis_task, send_ws_notification
from ..session_management.session_manager import SessionManager

logger = logging.getLogger(__name__)

class AnalysisAgent:
    """
    The Centralized Orchestrator for the ChatFlow AI agent.
    Role: Gathers intent, corrects parameters, and dispatches to Celery.
    Prevents LLM from directly controlling tool execution.
    """
    def __init__(self, session_id: str):
        self.session = AnalysisSession.objects.get(id=session_id)
        self.dataset = self.session.dataset

    def process_query(self, query: str) -> List[str]:
        """
        Processes a natural language query with context-awareness and real-time status.
        """
        # --- Persistence Layer (Bug Fix: Bug 12.4.1) ---
        # Record the user's intent in the persistent chat history.
        # We use get_or_create to prevent duplicates on rapid refreshes or retries.
        if not self.session.messages.filter(message_type='user', content=query).exists():
            SessionManager.add_user_message(self.session, query)

        # 0. Fetch Context (Last 10 messages for stateful analysis)
        history = list(self.session.messages.all().order_by('-created_at')[:10])
        history_data = [
            {"role": "user" if m.message_type == "user" else "assistant", "content": m.content}
            for m in reversed(history)
        ]

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

        # 2. Planning (LLM Intent Detection with History)
        intents = analysis_planner.create_plan(query, schema_text, history=history_data)
        if not intents:
            logger.warning(f"No intents generated for query: '{query}'")
            send_ws_notification(
                "I couldn't determine a plan. Please try rephrasing your request.",
                status="error",
                session_id=str(self.session.id)
            )
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
            
            # --- PARAMETER NORMALIZATION LAYER ---
            # LLM sometimes returns parameters as a list: [{"name": "col", "value": "Age"}]
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
            
            # A. Whitelist Validation
            tool_instance = tool_registry.get_tool(tool_name)
            if not tool_instance:
                logger.warning(f"Tool '{tool_name}' blocked by registry whitelist.")
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

# Accessor factory
def get_analysis_agent(session_id: str):
    return AnalysisAgent(session_id)
