"""
Mosaic Plot Tool
Visualizes the relationship between two categorical variables (graphical contingency table).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic
from scipy.stats import chi2_contingency
from typing import Dict, Any, List, Optional

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType
from .plot_utils import PlotUtils

class MosaicPlotTool(BaseAnalysisTool):
    """Tool for generating Mosaic Plots."""

    @property
    def name(self) -> str:
        return "mosaic_plot"

    @property
    def description(self) -> str:
        return "Visualize the relationship between two categorical variables (graphical Chi-Square)."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name="mosaic_plot")
        params.add_parameter(
            ToolParameter(
                name="var1",
                # Checking other files, CATEGORICAL_COLUMN_SELECT is standard.
                parameter_type=ParameterType.CATEGORICAL_COLUMN_SELECT,
                label="Variable 1",
                description="First categorical variable.",
                required=True
            )
        )
        params.add_parameter(
            ToolParameter(
                name="var2",
                parameter_type=ParameterType.CATEGORICAL_COLUMN_SELECT,
                label="Variable 2",
                description="Second categorical variable.",
                required=True
            )
        )
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            # 1. Get parameters
            parameters = kwargs
            var1 = parameters.get("var1")
            var2 = parameters.get("var2")

            if not var1 or not var2:
                return {"status": "error", "summary": "Please select two variables."}
            
            if var1 == var2:
                return {"status": "error", "summary": "Please select two different variables."}

            df = self.load_dataset(columns=[var1, var2])
            df_clean = df.dropna()

            if df_clean.empty:
                 return {"status": "error", "summary": "No data remaining after removing missing values."}

            # 2. Statistics (Chi-Square)
            contingency_table = pd.crosstab(df_clean[var1], df_clean[var2])
            chi2, p, dof, expected = chi2_contingency(contingency_table)
            
            # 3. Visualization
            artifacts = []
            with PlotUtils.setup_plotting():
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Mosaic Plot
                # mosaic returns (fig, rects) dict
                mosaic(df_clean, [var1, var2], ax=ax, title=f"Mosaic Plot: {var1} vs {var2}")
                # Note: mosaic function handles axis labeling usually
                
                artifacts.append({
                    "type": "plot",
                    "id": "mosaic_plot",
                    "title": "Mosaic Plot",
                    "content": PlotUtils.fig_to_base64(fig)
                })
                plt.close(fig)

            # 4. Results
            sections = []
            
            summary_text = f"Analyzed relationship between '{var1}' and '{var2}'."
            summary_text += f"\nChi-Square Statistic: {chi2:.4f}, P-value: {p:.4f}."
            if p < 0.05:
                summary_text += " Relationship is Significant."
            else:
                summary_text += " Relationship is Not Significant."
            
            # Residual Analysis for Dynamic Interpretation
            # Residuals = (Observed - Expected) / sqrt(Expected)
            # Std res > 2 or < -2 are interesting.
            
            residuals = (contingency_table - expected) / np.sqrt(expected)
            
            significant_assocs = []
            for r_idx in residuals.index:
                for c_idx in residuals.columns:
                    res_val = residuals.loc[r_idx, c_idx]
                    if abs(res_val) > 2:
                        direction = "over-represented" if res_val > 0 else "under-represented"
                        significant_assocs.append(f"- **{r_idx}** / **{c_idx}**: {direction} (Residual: {res_val:.2f})")
            
            if significant_assocs:
                summary_text += "\n\n### Key Drivers (Significant Residuals > 2):\n" + "\n".join(significant_assocs)
            else:
                 summary_text += "\n\n### Interpretation:\nNo specific category combinations showed unusually high or low frequencies (all residuals < 2)."
                
            sections.append({
                'type': 'text',
                'title': 'Chi-Square Test',
                'content': f"Chi2: {chi2:.4f}, p-value: {p:.4f}, dof: {dof}"
            })

            sections.append({
                'type': 'table',
                'title': 'Contingency Table (Observed)',
                'headers': [''] + list(map(str, contingency_table.columns)),
                'data': [[str(idx)] + list(map(str, row)) for idx, row in zip(contingency_table.index, contingency_table.values)]
            })

            return {
                "status": "ok",
                "summary": summary_text,
                "sections": sections,
                "artifacts": artifacts,
                "meta": {
                    "tool_name": self.name,
                    "parameters": parameters,
                }
            }

        except Exception as e:
            self.log_error(e)
            return {"status": "error", "summary": f"An unexpected error occurred: {str(e)}"}

    def interpret(self, results: Dict[str, Any]) -> Optional[str]:
        if results.get('status') != 'ok':
            return None
        return results.get('summary', "Mosaic Plot Completed.")