import logging
import json
from typing import List, Dict, Any, Optional
from ..llm_management.gemini_service import gemini_service
from ..analysis_tools.registry import tool_registry

logger = logging.getLogger(__name__)

class AnalysisPlanner:
    """
    The orchestrator that maps User Query → Tool Intents.
    Uses Gemini 1.5 with a strict 'Viz-Ready' tool-calling prompt.
    """
    def __init__(self, model_id: str = "gemini-1.5-flash"):
        self.model_id = model_id

    def create_plan(self, query: str, schema_text: str) -> List[Dict[str, Any]]:
        """
        Maps a query to a list of tool intents.
        Ex: 'Show correlation between price and rating' -> [{"tool": "correlation_matrix", "params": {"variables": ["price", "rating"]}}]
        """
        # 1. Get available tools metadata
        tools_meta = tool_registry.get_all_tool_metadata()
        tools_description = json.dumps(tools_meta, indent=2)

        # 2. Construct System Instruction
        system_instruction = f"""
        You are an expert data analyst planner for the ChatFlow system.
        Your task is to identify which analysis tools are needed to satisfy the user's query.
        
        DATASET SCHEMA (Columns & Types):
        {schema_text}
        
        AVAILABLE TOOLS:
        {tools_description}
        
        RESPONSE RULES:
        1. Return a JSON array of 'ToolIntent' objects.
        2. Format: [{"tool": "tool_name", "params": {"param_name": "value"}}]
        3. Use ONLY tool names from the whitelisted list above.
        4. Extract parameters exactly from the user query. If a column name is not exact, use the closest guess.
        5. If multiple tools are needed (e.g. 'stats and correlation'), return all of them in sequence.
        """

        prompt = f"User Query: '{query}'\nIdentify the tools and parameters."

        try:
            # 3. Extraction via Gemini (With JSON Response Mode)
            intents = gemini_service.get_intent_json(prompt, system_instruction)
            
            if isinstance(intents, dict) and "tool" in intents:
                intents = [intents] # Handle single intent object
            
            if not isinstance(intents, list):
                logger.error(f"Invalid intent format returned: {intents}")
                return []

            return intents
            
        except Exception as e:
            logger.error(f"Analysis Planning failed: {e}")
            return []

# Global accessor
analysis_planner = AnalysisPlanner()
