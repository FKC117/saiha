import logging
import pandas as pd
from django.conf import settings
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from alaina.celery import app
from .base import BaseAnalysisTask
from ..models import AnalysisSession, AnalysisResult, Dataset
from ..analysis_tools.registry import tool_registry
from ..agents.interpretation_agent import interpretation_agent

logger = logging.getLogger(__name__)

def send_ws_notification(message, status="info", task_id=None, session_id=None):
    """
    Helper to send real-time notification via Channels.
    Wrapped in try-except to prevent UI/Network issues from crashing Celery tasks.
    """
    try:
        channel_layer = get_channel_layer()
        if channel_layer and session_id:
            async_to_sync(channel_layer.group_send)(
                f"notification_{str(session_id)}",
                {
                    "type": "send_notification",
                    "message": message,
                    "status": status,
                    "task_id": str(task_id) if task_id else None,
                    "session_id": str(session_id) if session_id else None
                }
            )
    except Exception as e:
        logger.warning(f"Could not send WebSocket notification: {e}")

@app.task(base=BaseAnalysisTask, bind=True, name="saiha.celery_tasks.execute_analysis_task")
def execute_analysis_task(self, result_id: str, session_id: str, tool_name: str, 
                          params: dict, dedup_id: str, query: str = ""):
    """
    The Hardened Async Executor.
    """
    try:
        # Start Observability: Let the system know we are actually running
        self.start_observability(result_id, self.request.id)

        send_ws_notification(
            f"Executing analysis: {tool_name.replace('_', ' ').title()}...",
            status="executing",
            task_id=self.request.id,
            session_id=session_id
        )

        # 1. Fetch Context (Pass-by-Reference)
        session = AnalysisSession.objects.get(id=session_id)
        dataset = session.dataset
        
        # 2. Initialize Tool from Registry (Security Whitelist)
        tool = tool_registry.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool {tool_name} not found in Whitelist.")
        
        # Inject standard context for legacy tools
        tool.session = session
        tool.dataset = dataset
        tool.user = session.user

        # 3. Load Data (Hardened Column Extraction - Bug 13.1)
        dataset_columns = set(dataset.columns.values_list('column_name', flat=True))
        raw_relevant = []
        for v in params.values():
            if isinstance(v, str): raw_relevant.append(v)
            elif isinstance(v, list): raw_relevant.extend([x for x in v if isinstance(x, str)])
        
        # Whitelist Filtering: Only load strings that are actually valid columns
        relevant_cols = [c for c in raw_relevant if c in dataset_columns]
        
        df = tool.load_dataset(columns=list(set(relevant_cols)) if relevant_cols else None)
        
        # --- Memory Cache Optimization (Bug 12) ---
        # We pass the pre-filtered DF into the tool instance to prevent redundant I/O
        tool._df = df

        # 4. Execute Analysis
        result_obj = tool.validate_and_run(df, params)

        if result_obj.status == "error":
            raise Exception(result_obj.error or "Analysis failed without specific error.")

        # 5. Persist Results (Elite v3.8 Synchronization)
        # The result_record exists (created by Agent)
        result_record = AnalysisResult.objects.get(id=result_id)

        # 3-Stage Normalization (Sections -> Artifacts -> Fallback Data)
        # Ensuring the UI always receives its payload in the 'artifacts' key.
        final_artifacts = result_obj.artifacts or []
        
        # A. Section Scan
        if hasattr(result_obj, 'sections') and result_obj.sections:
            for section in result_obj.sections:
                final_artifacts.append(tool.sanitize_json_data(section))
        elif isinstance(result_obj.data, dict) and 'sections' in result_obj.data:
             for section in result_obj.data['sections']:
                final_artifacts.append(tool.sanitize_json_data(section))
        
        # B. Data Promotion (Promote tables/records from 'data' if artifacts are still sparse)
        if not final_artifacts and isinstance(result_obj.data, dict):
            for key, val in result_obj.data.items():
                if isinstance(val, dict) and ('records' in val or 'data' in val):
                    records = val.get('records') or val.get('data') or []
                    if records:
                        final_artifacts.append(tool.sanitize_json_data({
                            "type": "table",
                            "title": f"Data - {key.replace('_', ' ').title()}",
                            "data": [list(r.values()) if isinstance(r, dict) else r for r in records],
                            "headers": list(records[0].keys()) if isinstance(records[0], dict) else []
                        }))

        # JSON Hardening
        safe_result = tool.sanitize_json_data({
            "data": result_obj.data,
            "artifacts": final_artifacts,
            "message": result_obj.message
        })

        result_record.result_data = safe_result
        result_record.summary = result_obj.message 
        result_record.status = AnalysisResult.Status.SUCCESS
        result_record.save() # --- PERSIST BEFORE INTERPRETATION ---

        # 6. Notify: Analysis complete, moving to narrative generation
        send_ws_notification(
            "Analysis complete. Generating narratives and business insights...",
            status="generating",
            session_id=session_id,
            task_id=self.request.id
        )

        # 7. Trigger Interpreter (Post-Execution Analysis)
        # Pass final_artifacts to avoid redundant DB hits/normalization
        interpretation_agent.interpret_result(str(result_record.id), final_artifacts=final_artifacts)

        # 7. Notify Success
        send_ws_notification(
            f"{tool_name} analysis complete.", 
            status="success", 
            task_id=self.request.id,
            session_id=session_id
        )

        return {
            "result_id": str(result_record.id),
            "tool": tool_name,
            "artifacts_count": len(result_obj.artifacts)
        }

    except Exception as e:
        logger.error(f"Task Failed: {tool_name} - {str(e)}", exc_info=True)
        # BaseAnalysisTask will catch this and update the record status to FAILED
        raise
