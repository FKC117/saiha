import logging
import json
from typing import List, Dict, Any, Optional
from django.utils import timezone
from ..models import AnalysisSession, AnalysisResult, Dataset
from ..llm_management.gemini_service import gemini_service

logger = logging.getLogger(__name__)

class ReportBuilder:
    """
    The Intelligence Layer for Professional Exports.
    1. Scores results by 'Value' (Charts, Significance, Complexity).
    2. Generates Narrative Titles (LLM).
    3. Generates 3-Bullet Takeaways (LLM).
    4. Orchestrates Executive Summary.
    """
    def __init__(self, session_id: str):
        self.session = AnalysisSession.objects.get(id=session_id)
        self.dataset = self.session.dataset

    def _is_high_value(self, result: AnalysisResult) -> int:
        """
        Scores a result (0-100).
        Priority: Correlations (90), ANOVA/Tests (80), Distributions (60), Raw Tables (20).
        """
        score = 0
        data = result.result_data or {}
        artifacts = data.get('artifacts', [])
        
        # 1. Chart presence = High Value
        if any(a.get('type') == 'chart' for a in artifacts):
            score += 50
        
        # 2. Tool Type Weighting
        tool_used = result.tool_used.lower()
        if 'correlation' in tool_used: score += 40
        if 'test' in tool_used or 'anova' in tool_used: score += 35
        if 'descriptive' in tool_used: score += 20
        
        # 3. Data Richness
        raw_data = data.get('data', {})
        if len(raw_data) > 0: score += 10
        
        return score

    def _generate_insight_metadata(self, result: AnalysisResult) -> Dict[str, Any]:
        """
        Calls Gemini to generate [Title] and [3 Takeaways] for a specific result.
        """
        tool_used = result.tool_used
        data_json = json.dumps(result.result_data.get('data', {}))
        
        system_instruction = f"""
        You are a senior consulting analyst. Review the output of the '{tool_used}' tool.
        
        TASK:
        1. Generate a short, punchy 'Insight-First' title (e.g., 'Price Trends Show Right Skew').
        2. Provide exactly 3 bullet points of 'Key Takeaways' that explain what this means for a business.
        
        RULES:
        - Be specific.
        - Do not be speculative.
        - Output JSON: {{"title": "...", "takeaways": ["...", "...", "..."]}}
        """
        
        prompt = f"TOOL DATA: {data_json}"
        
        try:
            response = gemini_service.get_intent_json(prompt, system_instruction)
            return response if isinstance(response, dict) else {"title": tool_used, "takeaways": []}
        except Exception:
            return {"title": tool_used, "takeaways": []}

    def build_narrative_context(self, threshold: int = 40) -> Dict[str, Any]:
        """
        Orchestrates the entire report structure.
        """
        results = AnalysisResult.objects.filter(session=self.session, status="SUCCESS").order_by('created_at')
        
        high_value_insights = []
        for res in results:
            if self._is_high_value(res) >= threshold:
                meta = self._generate_insight_metadata(res)
                high_value_insights.append({
                    "id": str(res.id),
                    "tool": res.tool_used,
                    "title": meta.get('title', res.tool_used),
                    "takeaways": meta.get('takeaways', []),
                    "data": res.result_data.get('data', {}),
                    "artifacts": res.result_data.get('artifacts', [])
                })

        # Generate Executive Summary for the session
        summary_prompt = f"Summarize the following findings from {len(high_value_insights)} analyses into a consulting executive summary."
        summary_context = json.dumps([{"title": i['title'], "takeaways": i['takeaways']} for i in high_value_insights])
        
        exec_summary = gemini_service.generate_content(
            f"Findings: {summary_context}",
            "You are a Senior Analyst. Write 3-5 punchy summary bullet points for an 'Executive Summary' slide."
        ).split('\n')
        exec_summary = [s.strip('- ').strip('* ') for s in exec_summary if s.strip()]

        return {
            "session_title": f"Analytical Report: {self.dataset.name}",
            "date": timezone.now().strftime("%Y-%m-%d"),
            "dataset_info": {
                "name": self.dataset.name,
                "rows": self.dataset.rows_count,
                "cols": self.dataset.columns_count,
                "fields": [c.column_name for c in self.dataset.columns.all()[:5]]
            },
            "executive_summary": exec_summary[:5],
            "insights": high_value_insights
        }
