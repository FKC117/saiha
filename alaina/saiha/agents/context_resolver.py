import logging
import json
from typing import Dict, Any, Optional
from ..models import AnalysisSession

logger = logging.getLogger(__name__)

class ContextResolver:
    """
    The Authority Layer (Elite Mode v3.2+).
    Resolves conflicting memory signals into a single "Source of Truth".
    Enforces strict hard limits on context size to prevent explosion.
    """
    
    # Production Hard Limits
    SUMMARY_MAX_CHARS = 1000
    ANALYSIS_CHAIN_MAX = 5
    ACTIVE_COLUMNS_MAX = 5
    RECENT_MESSAGES_MAX = 3

    @staticmethod
    def is_complete(meta: Dict[str, Any]) -> bool:
        """
        Strictly verify if a metadata signal is functionally complete.
        Requires both 'active_columns' and 'last_result_type'.
        """
        required = ["active_columns", "last_result_type"]
        return all(meta.get(k) for k in required)

    @staticmethod
    def resolve_state(session: AnalysisSession) -> Dict[str, Any]:
        """
        Unifies all memory layers with a Quality-Aware priority:
        1. Last Valid Metadata (ONLY if Complete)
        2. Working Memory (Active UI/Conversation state)
        3. Rolling Summary (Long-term reasoning context)
        """
        # Start with Summary as base
        state = {
            "summary": (session.memory_summary or "")[:ContextResolver.SUMMARY_MAX_CHARS],
            "active_columns": [],
            "last_tool": None,
            "last_result_type": None,
            "analysis_chain": (session.analysis_chain or [])[:ContextResolver.ANALYSIS_CHAIN_MAX],
            "source": "summary"
        }

        # Fetch candidates
        wm = session.working_memory or {}
        meta = session.last_valid_metadata or {}

        # Layer 1: Working Memory (Good default for specificity)
        if wm.get("active_columns"):
            state["active_columns"] = wm.get("active_columns")[:ContextResolver.ACTIVE_COLUMNS_MAX]
            state["source"] = "working_memory"
        
        if wm.get("last_tool"): state["last_tool"] = wm.get("last_tool")
        if wm.get("last_result_type"): state["last_result_type"] = wm.get("last_result_type")

        # Layer 2: Last Valid Metadata (Override ONLY if complete)
        if meta and ContextResolver.is_complete(meta):
            state["active_columns"] = meta.get("active_columns")[:ContextResolver.ACTIVE_COLUMNS_MAX]
            state["last_result_type"] = meta.get("last_result_type")
            state["last_target_column"] = meta.get("last_target_column")
            state["source"] = "metadata"
        elif meta:
            logger.warning("Metadata was partial; deferring to Working Memory/Summary.")

        return state

    @staticmethod
    def format_resolved_state(state: Dict[str, Any]) -> str:
        """
        Formats the resolved state into a single, clean block for the LLM.
        Avoids redundant layers to reduce "attention drift".
        """
        summary_text = f"\n### CONTEXT SUMMARY\n{state['summary']}" if state['summary'] else ""
        
        return f"""{summary_text}

### CURRENT CONTEXT (Source: {state['source']})
- **Active Columns**: {", ".join(state['active_columns']) if state['active_columns'] else "None"}
- **Last Analysis**: {state['last_result_type'] or "N/A"}
- **Previous Steps**: {" -> ".join(state['analysis_chain']) if state['analysis_chain'] else "None"}
"""
