import json
import logging
import pandas as pd
import numpy as np
from .base_tool import BaseAnalysisTool
import re
import time
from django.db import models
from .tool_parameters import ToolParameterSet
from ..models import Tool, AnalysisResult

logger = logging.getLogger(__name__)

# ... (omitted code) ...



logger = logging.getLogger(__name__)

class SmartRecommendationsTool(BaseAnalysisTool):
    """
    A tool that uses an LLM to recommend other analysis tools based on the
    structure and metadata of the current dataset.
    """
    name = "Smart Recommendations"
    tool_type = "smart_recommendations"
    description = "Get AI-powered analysis recommendations based on your dataset's structure."

    def get_parameters_schema(self) -> ToolParameterSet:
        """
        This tool's parameters are defined in the central tool_parameters.py file.
        """
        return ToolParameterSet(tool_name=self.name)

    def _calculate_lightweight_profile(self, df: pd.DataFrame) -> list:
        """
        Calculates a lightweight statistical profile for the dataset, specifically designed
        to give the LLM context about distributions (skew, outliers) without token overload.
        """
        profile = []
        for col in df.columns:
            col_data = df[col]
            # Basic info
            dtype = str(col_data.dtype)
            col_stats = {
                'column_name': str(col),
                'data_type': dtype,
                'null_percent': round((col_data.isnull().sum() / len(df)) * 100, 1),
                'unique_count': int(col_data.nunique())
            }
            
            # Numeric stats (Distributions)
            if pd.api.types.is_numeric_dtype(col_data):
                try:
                    # Drop NA for stats
                    clean_data = col_data.dropna()
                    if not clean_data.empty:
                        col_stats.update({
                            'mean': round(float(clean_data.mean()), 2),
                            'std': round(float(clean_data.std()), 2),
                            'min': round(float(clean_data.min()), 2),
                            'max': round(float(clean_data.max()), 2),
                            'skew': round(float(clean_data.skew()), 2),
                        })
                        # Kurtosis can be unstable with few samples
                        if len(clean_data) > 3:
                            col_stats['kurtosis'] = round(float(clean_data.kurtosis()), 2)
                except Exception as e:
                    logger.warning(f"Error calculating numeric stats for {col}: {e}")

            # Categorical stats (Examples)
            elif pd.api.types.is_object_dtype(col_data) or pd.api.types.is_categorical_dtype(col_data):
                try:
                    # Add top 3 common values for context
                    top_values = col_data.value_counts().head(5).index.tolist()
                    col_stats['sample_values'] = [str(v) for v in top_values]
                except Exception:
                    col_stats['sample_values'] = []
            
            profile.append(col_stats)
        return profile

    def execute(self, query: str = "", **kwargs) -> dict:
        """
        Executes the smart recommendation logic.
        """
        self.validate_dataset_requirement()
        logger.info(f"Executing SmartRecommendationsTool for dataset: {self.dataset.name if self.dataset else 'N/A'}")

        # 1. Load the dataframe to calculate fresh, detailed statistics
        df = self.load_dataset()
        
        # 2. detailed column summary for the dataset
        columns_summary = self._calculate_lightweight_profile(df)

        # NEW: Fetch Lineage History
        lineage_summary = []
        if self.dataset:
            try:
                # Get all ancestor sessions
                lineage_sessions = self.dataset.get_lineage_sessions()
                # Get results from these sessions, ordered chronologically
                past_results = AnalysisResult.objects.filter(session__in=lineage_sessions).select_related('session').order_by('created_at')
                
                for res in past_results:
                    # Summarize each past action: Tool Name + AI Interpretation (or summary)
                    action_summary = {
                        'tool': res.tool_used,
                        'date': res.created_at.strftime("%Y-%m-%d %H:%M"),
                        'summary': res.pptx_summary or res.ai_interpretation[:200] + "..." if res.ai_interpretation else "No summary available."
                    }
                    lineage_summary.append(action_summary)
            except Exception as e:
                logger.warning(f"Error fetching lineage for recommendations: {e}")

        # 3. Get the list of available tools, excluding this one
        # Fetch detailed info to provide codes to the LLM
        # We use the 'tool_type' field from the DB which acts as the unique code/ID.
        # This matches the tool_registry keys (e.g. 'pca', 'lasso_ridge_regression').
        tools_qs = Tool.objects.filter(is_active=True).exclude(tool_type=self.tool_type)
        available_tools = [f"{t.name} (ID: {t.tool_type})" for t in tools_qs]

        # 4. Construct the prompt for the LLM
        prompt = self._build_prompt(columns_summary, available_tools, lineage_summary)


        # 5. Call the LLM (Using the new hardened GeminiService)
        from ..llm_management.gemini_service import gemini_service
        
        llm_response_str = ""
        pptx_summary = ""
        recommendations_json = []

        try:
            start_time = time.time()
            llm_full_response = gemini_service.generate_content(prompt, session_id=str(self.session.id), user=self.session.user)
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            # Parse pptx summary
            match_pptx = re.search(r'<pptx_summary>(.*?)</pptx_summary>', llm_full_response, re.DOTALL)
            if match_pptx:
                pptx_summary = match_pptx.group(1).strip()
                llm_response_str = llm_full_response.replace(match_pptx.group(0), '').strip()
            else:
                llm_response_str = llm_full_response

            # Parse action buttons JSON
            match_json = re.search(r'<action_buttons>(.*?)</action_buttons>', llm_full_response, re.DOTALL)
            if match_json:
                try:
                    json_str = match_json.group(1).strip()
                    # Clean potential markdown code blocks inside the tag
                    json_str = json_str.replace('```json', '').replace('```', '')
                    raw_recommendations = json.loads(json_str)
                    
                    # Validate tools exist in the available set
                    # We utilize the 'tool_type' field from the database as the source of truth for IDs.
                    valid_tool_codes = set(
                         Tool.objects.filter(is_active=True)
                         .exclude(tool_type=self.tool_type)
                         .values_list('tool_type', flat=True)
                    )
                    
                    recommendations_json = []
                    for rec in raw_recommendations:
                        tool_code = rec.get('tool_name')
                        if tool_code in valid_tool_codes:
                            recommendations_json.append(rec)
                        else:
                            logger.warning(f"Skipping recommended tool '{tool_code}' as it is not a valid registered tool ID.")

                    # Remove the JSON tag from the display text
                    llm_response_str = llm_response_str.replace(match_json.group(0), '').strip()
                except Exception as e:
                    logger.warning(f"Failed to parse action buttons JSON: {e}")

            logger.debug(f"Raw LLM response for Smart Recommendations: '{llm_response_str}'")
        except Exception as e:
            logger.error(f"Error calling LLM for Smart Recommendations: {e}")
            return {'status': 'error', 'summary': 'Failed to get a response from the AI model.'}

        # 6. Format the result for the template
        return {
            'status': 'ok',
            'summary': f"AI-powered recommendations for the '{self.dataset.name}' dataset.",
            'sections': [{
                'type': 'smart_recommendations', # Use a dedicated type for special rendering.
                'title': 'AI-Powered Tool Recommendations',
                'icon': 'fas fa-lightbulb',
                'content': llm_response_str,
                'actions': recommendations_json # Pass actions to the template
            }],
            'pptx_summary': pptx_summary  # Pass this back to the view
        }

    def _build_prompt(self, columns_summary: list, available_tools: list, lineage_summary: list = None) -> str:
        """Constructs the detailed prompt for the LLM, including lineage history."""
        
        history_context = ""
        if lineage_summary:
            history_context = f"""
        **DATASET HISTORY (LINEAGE):**
        This dataset has evolved through the following steps. CONSIDER THIS HISTORY when making recommendations.
        For example, if outliers were just removed, do not recommend outlier treatment again. if the data was just transformed, suggest modeling.
        ```json
        {json.dumps(lineage_summary, indent=2)}
        ```
            """
        
        return f"""
        You are an expert data scientist acting as a helpful recommender system. A user is planning their analysis on a dataset, and your task is to suggest the most relevant analysis tools to uncover valuable insights.

        {history_context}

        **DATASET CONTEXT:**
        Here is a detailed summary of the dataset's columns:
        ```json
        {json.dumps(columns_summary, indent=2)}
        ```

        **AVAILABLE TOOLS:**
        Here is a list of the analysis tools available to the user:
        {json.dumps(available_tools)}

        **YOUR TASK:**
        1. Analyze the dataset context provided. **Pay special attention to 'skew' and 'kurtosis' values to identify non-normal distributions, and 'null_percent' for missing data.**
        2. Design a comprehensive **Analysis Workflow** to uncover valuable insights. Do not limit the number of tools; suggest as many as necessary to conclude a robust analysis.
        3. Structure your response in logical **Phases** (e.g., Phase 1: Data Preparation, Phase 2: Exploratory Analysis, Phase 3: Modeling).
        4. **CRITICAL - DATA PERSISTENCE LOGIC**:
            *   Tools like **"Variable Transformation"**, **"Outlier Treatment"**, **"Missing Value Imputation"**, **"Recode Variable"**, **"Filter Rows"**, and **"Compute Variable"** act as *Data Modifiers*.
            *   When you recommend a *Data Modifier*, explicitly state that it will **save a NEW dataset**.
            *   You MUST instruct the user to **load this new dataset** before proceeding to subsequent steps (like PCA, Regression, or ANOVA) to ensure they analyze the cleaned/transformed data.
        5. For each recommended tool, provide a brief explanation of **why** it is chosen for this specific dataset and **what** specific columns to target.

        **OUTPUT FORMAT REQUIREMENTS:**
        
        A. **Narrative**: Provide the detailed textual recommendation as described above.

        B. **Action Buttons**: At the end of your response, provide a JSON list of actionable buttons strictly within `<action_buttons>` tags.
        Each item must have:
        - `tool_name`: The **ID** of the tool (e.g. `variable_transformation` NOT "Variable Transformation"). Look at the "AVAILABLE TOOLS" list for IDs.
        - `label`: Short action label (e.g., "Run PCA", "Log Transform").
        - `reason`: Very short reason (max 10 words).
        - `params_hint`: A dictionary of suggested parameters (if applicable, e.g., target columns).

        Example Action Buttons:
        <action_buttons>
        [
            {{
                "tool_name": "variable_transformation",
                "label": "Log Transform Income",
                "reason": "Fix high skewness (1.5)",
                "params_hint": {{ "columns": ["Income"], "method": "log" }}
            }},
            {{
                "tool_name": "kmeans_clustering",
                "label": "Cluster Customers",
                "reason": "Segment based on spending",
                "params_hint": {{ "n_clusters": 3 }}
            }}
        ]
        </action_buttons>

        C. **PPTX Summary**: Finally, provide a separate, concise summary of the workflow for a PowerPoint slide. Enclose this simple summary in `<pptx_summary>` tags.
        """

    def interpret(self, result: dict) -> str:
        return "The recommendations above are based on your dataset's structure. Click on a tool name to start the analysis."