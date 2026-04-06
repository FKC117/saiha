"""
Reliability Analysis Tool
Performs reliability analysis (Cronbach's Alpha) on a set of items.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType
from .plot_utils import PlotUtils

class ReliabilityAnalysisTool(BaseAnalysisTool):
    """Tool for performing Reliability Analysis (Cronbach's Alpha)."""

    @property
    def name(self) -> str:
        return "reliability_analysis"

    @property
    def description(self) -> str:
        return "Calculate Cronbach's Alpha to assess the internal consistency reliability of a set of items."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name="reliability_analysis")
        params.add_parameter(
            ToolParameter(
                name="numeric_columns",
                parameter_type=ParameterType.MULTISELECT,
                label="Select Items (Numeric Columns)",
                description="Choose the numeric columns (items) to include in the scale.",
                required=True,
                column_source="numeric"
            )
        )
        return params

    def calculate_cronbach_alpha(self, df: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
        """
        Calculates Cronbach's Alpha and item-total statistics.
        """
        # 1. Overall Cronbach's Alpha
        item_scores = df
        item_variances = item_scores.var(axis=0, ddof=1)
        total_score_variance = item_scores.sum(axis=1).var(ddof=1)
        n_items = item_scores.shape[1]
        
        if n_items < 2:
            return 0.0, pd.DataFrame()

        alpha = (n_items / (n_items - 1)) * (1 - (item_variances.sum() / total_score_variance))
        
        # 2. Item-Total Statistics (Alpha if deleted)
        item_stats = []
        total_scores = item_scores.sum(axis=1)
        
        for col in item_scores.columns:
            # Calculate correlation between item and total score (corrected for item overlap)
            # Corrected item-total correlation: Correlation between item and (Total - Item)
            rest_scores = total_scores - item_scores[col]
            corrected_item_total_corr = item_scores[col].corr(rest_scores)
            
            # Calculate Alpha if item deleted
            reduced_df = item_scores.drop(columns=[col])
            reduced_variances = reduced_df.var(axis=0, ddof=1)
            reduced_total_variance = reduced_df.sum(axis=1).var(ddof=1)
            n_reduced = n_items - 1
            
            if n_reduced < 2:
                alpha_if_deleted = np.nan
            else:
                alpha_if_deleted = (n_reduced / (n_reduced - 1)) * (1 - (reduced_variances.sum() / reduced_total_variance))
            
            item_stats.append({
                'Item': col,
                'Corrected Item-Total Correlation': corrected_item_total_corr,
                'Cronbach\'s Alpha if Item Deleted': alpha_if_deleted
            })
            
        return alpha, pd.DataFrame(item_stats)

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            # 1. Get parameters and load data
            parameters = kwargs
            numeric_columns = parameters.get("numeric_columns", [])
            if isinstance(numeric_columns, str):
                numeric_columns = [numeric_columns]

            if not numeric_columns or len(numeric_columns) < 2:
                return {"status": "error", "summary": "Please select at least 2 items (columns) for reliability analysis."}

            df = self.load_dataset(columns=numeric_columns)
            
            # Handle missing values (listwise deletion is standard for reliability)
            df_clean = df.dropna()
            if df_clean.empty:
                return {"status": "error", "summary": "Dataset is empty after removing missing values."}
            
            # 2. Calculate Alpha
            overall_alpha, item_stats_df = self.calculate_cronbach_alpha(df_clean)

            artifacts = []
            sections = []

            # 3. Overall Reliability Table
            sections.append({
                'type': 'table',
                'title': 'Reliability Statistics',
                'icon': 'bi bi-check-circle',
                'headers': ['Cronbach\'s Alpha', 'N of Items'],
                'data': [[f"{overall_alpha:.4f}", len(numeric_columns)]]
            })
            
            # 4. Item-Total Statistics Table
            if not item_stats_df.empty:
                # Format floats
                formatted_stats = item_stats_df.copy()
                for col in ['Corrected Item-Total Correlation', 'Cronbach\'s Alpha if Item Deleted']:
                    formatted_stats[col] = formatted_stats[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
                
                sections.append({
                    'type': 'table',
                    'title': 'Item-Total Statistics',
                    'icon': 'bi bi-list-check',
                    'headers': formatted_stats.columns.tolist(),
                    'data': formatted_stats.values.tolist()
                })

            # 5. Inter-Item Correlation Matrix Heatmap
            corr_matrix = df_clean.corr()
            
            with PlotUtils.setup_plotting():
                fig, ax = plt.subplots(figsize=(max(8, len(numeric_columns)*0.8), max(6, len(numeric_columns)*0.6)))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax, vmin=-1, vmax=1)
                ax.set_title('Inter-Item Correlation Matrix')
                plt.tight_layout()
                
                artifacts.append({
                    "type": "plot",
                    "id": "correlation_heatmap",
                    "title": "Inter-Item Correlation Matrix",
                    "content": PlotUtils.fig_to_base64(fig)
                })
                plt.close(fig)

            # Interpretation
            interpretation = ""
            if overall_alpha >= 0.9:
                interpretation = "Excellent internal consistency."
            elif overall_alpha >= 0.8:
                interpretation = "Good internal consistency."
            elif overall_alpha >= 0.7:
                interpretation = "Acceptable internal consistency."
            elif overall_alpha >= 0.6:
                interpretation = "Questionable internal consistency."
            elif overall_alpha >= 0.5:
                interpretation = "Poor internal consistency."
            else:
                interpretation = "Unacceptable internal consistency."

            summary = f"Reliability analysis performed on {len(numeric_columns)} items. Cronbach's Alpha is {overall_alpha:.4f} ({interpretation})."
            
            # Dynamic Item Analysis
            improvement_suggestions = []
            if not item_stats_df.empty:
                for idx, row in item_stats_df.iterrows():
                    item_name = row['Item']
                    alpha_if_del = row['Cronbach\'s Alpha if Item Deleted']
                    
                    # If removing item improves alpha by > 0.05 (threshold)
                    if pd.notnull(alpha_if_del) and alpha_if_del > overall_alpha + 0.02: # 0.02 is a decent threshold
                         diff = alpha_if_del - overall_alpha
                         improvement_suggestions.append(f"- **{item_name}**: Removing this would increase Alpha to {alpha_if_del:.4f} (+{diff:.4f}).")
            
            if improvement_suggestions:
                summary += "\n\n### Optimization Suggestions:\n" + "\n".join(improvement_suggestions)
            else:
                summary += "\n\n### Item Analysis:\nNo items were found that would significantly improve reliability if removed."

            return {
                "status": "ok",
                "summary": summary,
                "sections": sections,
                "artifacts": artifacts,
                "meta": {
                    "tool_name": self.name,
                    "parameters": parameters,
                    "columns_analyzed": numeric_columns
                }
            }

        except Exception as e:
            self.log_error(e)
            return {"status": "error", "summary": f"An unexpected error occurred: {str(e)}"}

    def interpret(self, results: Dict[str, Any]) -> Optional[str]:
        if results.get('status') != 'ok':
            return None
        return results.get('summary', "Reliability Analysis Completed.")