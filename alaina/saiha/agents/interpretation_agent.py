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
    def __init__(self, model_id: str = "gemini-1.5-flash"):
        self.model_id = model_id

    def interpret_result(self, result_id: str) -> str:
        """
        Interprets a specific analysis result.
        """
        try:
            result = AnalysisResult.objects.get(id=result_id)
            if result.status != "SUCCESS":
                return "Analysis did not complete successfully; skipped interpretation."

            tool_name = result.tool_name
            data_json = result.result_data
            query = result.query

            # Build Interpretation Prompt
            system_instruction = f"""
            You are a senior data scientist. Interpret the following statistical result for a user.
            CONTEXT: The user asked '{query}' using the '{tool_name}' tool.
            
            RULES:
            1. Be concise but insightful.
            2. Explain what the numbers/charts mean in business terms.
            3. Highlight any anomalies (outliers, skewness, strong correlations).
            4. If no interesting findings exist, state that clearly.
            """

            prompt = f"TOOL OUTPUT: {data_json}"

            interpretation = gemini_service.generate_content(prompt, system_instruction)
            
            # Update the result record
            result.ai_interpretation = interpretation
            result.save()
            
            logger.info(f"Generated interpretation for Result {result_id}")
            return interpretation

        except Exception as e:
            logger.error(f"Interpretation failed for Result {result_id}: {e}")
            return f"Interpretation error: {str(e)}"

# Global accessor
interpretation_agent = InterpretationAgent()
