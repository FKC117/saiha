import logging
import pandas as pd
from django.conf import settings
from alaina.celery import app
from .base import BaseAnalysisTask
from ..models import AnalysisSession, AnalysisResult, Dataset
from ..analysis_tools.registry import tool_registry
from ..agents.interpretation_agent import interpretation_agent

logger = logging.getLogger(__name__)

@app.task(base=BaseAnalysisTask, bind=True, name="saiha.celery_tasks.execute_analysis_task")
def execute_analysis_task(self, session_id: str, tool_name: str, params: dict, 
                          dedup_id: str, query: str = ""):
    """
    The Hardened Async Executor.
    Runs the selected tool, persists results (ECharts/Tables), 
    and triggers the InterpretationAgent.
    """
    try:
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

        # 3. Load Data (Optimized via Universal Bridge)
        # We projection-load only the columns identified by the planner/corrector
        # (Heuristic: extract all strings from params that look like columns)
        relevant_cols = []
        for v in params.values():
            if isinstance(v, str): relevant_cols.append(v)
            elif isinstance(v, list): relevant_cols.extend([x for x in v if isinstance(x, str)])
        
        df = tool.load_dataset(columns=list(set(relevant_cols)) if relevant_cols else None)

        # 4. Execute Analysis (Universal Bridge: Hardened vs Legacy)
        # This handles Pydantic validation internally
        result_obj = tool.validate_and_run(df, params)

        if result_obj.status == "error":
            raise Exception(result_obj.error or "Analysis failed without specific error.")

        # 5. Persist Results (Viz-Ready)
        # We update the AnalysisResult record (The BaseAnalysisTask handles status=SUCCESS)
        # Note: AnalysisResult has unique=True on dedup_id
        # We try to get or create to ensure idempotency
        
        # We find the record (it might have been created by the Agent as PENDING)
        result_record, created = AnalysisResult.objects.get_or_create(
            dedup_id=dedup_id,
            defaults={
                "session": session,
                "tool_name": tool_name,
                "query": query,
                "status": "RUNNING"
            }
        )

        result_record.result_data = {
            "data": result_obj.data,
            "artifacts": result_obj.artifacts,
            "message": result_obj.message
        }
        result_record.task_id = self.request.id
        result_record.save()

        # 6. Trigger Interpreter (Post-Execution Analysis)
        # Interpreter runs in the same worker/thread for now (simple chaining)
        interpretation_agent.interpret_result(str(result_record.id))

        return {
            "result_id": str(result_record.id),
            "tool": tool_name,
            "artifacts_count": len(result_obj.artifacts)
        }

    except Exception as e:
        logger.error(f"Task Failed: {tool_name} - {str(e)}", exc_info=True)
        # BaseAnalysisTask will catch this and update the record status to FAILED
        raise
