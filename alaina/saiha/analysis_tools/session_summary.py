import json
import logging
import re
from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet
from saiha.models import AnalysisResult, Tool, LLMUsageLog
from saiha.dataset_utils import get_dataset_column_types
from saiha.ai_agents.config import get_model_config
from saiha.llm_service import generate_gemini_interpretation
import time

logger = logging.getLogger(__name__)

class SessionSummaryTool(BaseAnalysisTool):
    """
    A tool that synthesizes all analysis results from a session into a
    final session summary.
    """
    name = "Generate Session Summary"
    tool_type = "session_summary"
    description = "Generates a high-level summary of all analyses performed in the current session."

    def get_parameters_schema(self) -> ToolParameterSet:
        """This tool requires no parameters."""
        return ToolParameterSet(tool_name=self.name)

    def execute(self, query: str = "", **kwargs) -> dict:
        """
        Gathers all analysis results, constructs a prompt, and calls an LLM
        to generate a cohesive summary.
        """
        self.validate_dataset_requirement()
        logger.info(f"Executing SessionSummaryTool for session: {self.session.id}")

        # 1. --- CONTEXT GATHERING ---
        results = AnalysisResult.objects.filter(session=self.session).order_by('created_at')

        if results.count() < 2:
            return {'status': 'error', 'summary': 'At least two analysis results are needed to generate a meaningful summary.'}

        analysis_flow = []
        for result in results:
            tool_name = result.tool_used.replace('_', ' ').title()
            # Prioritize the highest quality summary available for each step
            if result.ai_interpretation:
                summary = result.ai_interpretation
            elif result.result_data.get('canned_interpretation'):
                summary = result.result_data['canned_interpretation']
            elif result.result_data.get('structured_result', {}).get('summary'):
                summary = result.result_data['structured_result']['summary']
            else:
                summary = "A data analysis step was performed."
            
            analysis_flow.append(f"### Step: {tool_name}\n{summary}\n")

        # 2. --- PROMPT CONSTRUCTION ---
        dataset_summary = get_dataset_column_types(self.dataset.id)
        chronological_summary = "\n---\n".join(analysis_flow)

        prompt = f"""
        You are an expert data analyst writing a session summary for a client.
        Your task is to synthesize the findings from a series of analyses into a single, cohesive narrative.

        **DATASET CONTEXT:**
        The analysis was performed on the '{self.dataset.name}' dataset.
        Column Structure: {json.dumps(dataset_summary, indent=2)}

        **ANALYSIS CHRONOLOGY & FINDINGS:**
        The following analyses were performed in order. Each section contains the findings from that step.
        {chronological_summary}

        **YOUR TASK:**
        Based on all the information above, write a final, high-level session summary. Structure your response in Markdown.
        1.  **Overall Summary:** Start with the most important conclusion drawn from the entire analysis journey.
        2.  **Key Findings:** List 2-3 of the most critical, data-supported insights from the different analysis steps.
        3.  **Final Recommendation:** Conclude with a clear, actionable recommendation for the user based on the synthesis of all results.

        **FINALLY, provide a separate, concise summary specifically for a PowerPoint slide.** This summary should be 3-4 short bullet points that capture the absolute most important takeaways from the entire session. Enclose this summary in `<pptx_summary>` tags. For example:
        <pptx_summary>
        - Key Finding: The primary driver of X was Y.
        - Actionable Insight: We should focus on Z to improve outcomes.
        </pptx_summary>

        Do not simply repeat the findings from each step. Your value is in connecting the dots and creating a holistic conclusion.
        """

        start_time = time.time()
        # 3. --- LLM INVOCATION & RESPONSE ---
        from ..llm_management.gemini_service import gemini_service
        from ..models import AIAuditLog
        
        start_time = time.time()
        try:
            llm_full_response = gemini_service.generate_response(prompt, session_id=str(self.session.id))
            execution_time_ms = int((time.time() - start_time) * 1000)

            # Log to the new Audit Trail
            AIAuditLog.objects.create(
                session=self.session,
                prompt=prompt,
                response=llm_full_response,
                model_id=gemini_service.model_id
            )
        except Exception as e:
            logger.error(f"LLM Synthesis failed in SessionSummaryTool: {e}")
            return {'status': 'error', 'summary': f"Detailed synthesis failed: {str(e)}"}

        # Parse the response for the main summary and the pptx summary
        pptx_summary = ""
        final_summary = llm_full_response
        match = re.search(r'<pptx_summary>(.*?)</pptx_summary>', llm_full_response, re.DOTALL)
        if match:
            pptx_summary = match.group(1).strip()

        execution_time_ms = int((time.time() - start_time) * 1000)

        # --- LOG THE USAGE ---
        # This log is linked to the session, not a specific analysis result.
        LLMUsageLog.objects.create(
            user=self.user,
            analysis_session=self.session,
            analysis_result=None, # This is a session-level summary
            prompt_text=prompt,
            response_text=final_summary,
            request_token_count=req_tokens,
            response_token_count=res_tokens,
            model_used=model_config.get('model', 'unknown'),
            execution_time_ms=execution_time_ms
        )
        logger.info(
            f"Session Summary LLM usage logged for user {self.user.id}. "
            f"Tokens (in/out): {req_tokens}/{res_tokens}. Time: {execution_time_ms}ms."
        )

        return {
            'status': 'ok',
            'summary': f"Session summary for the '{self.dataset.name}' analysis session.",
            'sections': [{
                'type': 'smart_recommendations', # Re-use the same section type for simple markdown rendering
                'title': 'Session Summary',
                'icon': 'fas fa-award',
                'content': final_summary.replace(match.group(0), '').strip() if match else final_summary
            }],
            'pptx_summary': pptx_summary  # Pass this back to the view
        }