"""
MANOVA Tool
Performs Multivariate Analysis of Variance (MANOVA) to test effects on multiple dependent variables.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from statsmodels.multivariate.manova import MANOVA
except ImportError:
    MANOVA = None

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType
from .plot_utils import PlotUtils

class ManovaTool(BaseAnalysisTool):
    """Tool for Multivariate ANOVA."""

    @property
    def name(self) -> str:
        return "manova_tool"

    @property
    def description(self) -> str:
        return "Test for effects of a categorical variable on multiple continuous outcome variables simultaneously."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name="manova_tool")
        params.add_parameter(
            ToolParameter(
                name="dependent_variables",
                parameter_type=ParameterType.MULTISELECT,
                label="Dependent Variables (Outcomes)",
                description="Select 2 or more numeric outcome variables.",
                required=True,
                column_source="numeric",
                validation_rules={"minItems": 2},
                help_text="Select at least 2 variables for multivariate analysis."
            )
        )
        params.add_parameter(
            ToolParameter(
                name="independent_variable",
                parameter_type=ParameterType.CATEGORICAL_COLUMN_SELECT,
                label="Independent Variable (Factor)",
                description="Grouping variable (categorical).",
                required=True
            )
        )
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            if MANOVA is None:
                return {"status": "error", "summary": "Statsmodels library is missing or limited. MANOVA is unavailable."}

            # 1. Get parameters
            parameters = kwargs
            dvs = parameters.get("dependent_variables", [])
            # Handle potential single string input
            if isinstance(dvs, str):
                dvs = [dvs]

            iv = parameters.get("independent_variable")
            iv = parameters.get("independent_variable")

            if not dvs or len(dvs) < 2 or not iv:
                return {"status": "error", "summary": "Please select at least 2 Dependent Variables and 1 Independent Variable."}

            # 2. Load and Prepare Data
            # Rename columns to safe names for formula compatibility
            df = self.load_dataset(columns=dvs + [iv])
            df_clean = df.dropna()

            if df_clean.empty:
                 return {"status": "error", "summary": "No data remaining after removing missing values."}

            # Sanitize column names for formula (MANOVA expects 'y1 + y2 ~ x')
            # Create mapping: original -> safe
            safe_iv = "independent_var"
            safe_dvs = [f"dependent_var_{i}" for i in range(len(dvs))]
            
            rename_map = {iv: safe_iv}
            for original, safe in zip(dvs, safe_dvs):
                rename_map[original] = safe
            
            df_safe = df_clean.rename(columns=rename_map)
            
            # Construct Formula
            formula = f"{' + '.join(safe_dvs)} ~ {safe_iv}"
            
            # 3. Fit MANOVA
            manova = MANOVA.from_formula(formula, data=df_safe)
            mv_test = manova.mv_test()
            
            # 4. Process Results
            artifacts = []
            sections = []
            
            # Extract Wilks' Lambda for the Independent Variable
            # mv_test.results is a dict keyed by term name (including 'Intercept')
            # We want the IV result.
            
            iv_result = mv_test.results.get(safe_iv)
            
            if iv_result is None:
                # Should not happen if fitted correctly
                summary_text = "Analysis ran but could not extract specific term results."
            else:
                # result['stat'] is the dataframe containing Lambda, Pillai, Hotelling, Roy
                stats_df = iv_result['stat']
                
                # Format for display
                # We specifically look for "Wilks' lambda" usually preferred
                wilks_row = stats_df.loc["Wilks' lambda"]
                lambda_val = wilks_row['Value']
                f_stat = wilks_row['F Value']
                p_val = wilks_row['Pr > F']
                
                summary_text = f"MANOVA Test for '{iv}' on {', '.join(dvs)}.\nWilks' Lambda: {lambda_val:.4f}, F-stat: {f_stat:.4f}, P-value: {p_val:.4f}."
                
                if p_val < 0.05:
                    summary_text += " Result is Significant (groups differ)."
                else:
                    summary_text += " Result is Not Significant."

                sections.append({
                    'type': 'table',
                    'title': 'Multivariate Test Statistics',
                    'headers': ['Statistic', 'Value', 'F Value', 'P-Value'],
                    'data': [
                        [idx, f"{row['Value']:.4f}", f"{row['F Value']:.4f}", f"{row['Pr > F']:.4f}"]
                        for idx, row in stats_df.iterrows()
                    ]
                })

                # Visualizations: Boxplots for each Dependent Variable
                # Generate separate image cards for each DV
                unique_groups = df_clean[iv].unique()
                palette = dict(zip(unique_groups, plt.cm.tab10.colors[:len(unique_groups)]))
                
                # Check for individual variable differences (simple ANOVA view) to enhance interpretation
                # We won't run full ANOVA but we can check means.
                group_means = df_clean.groupby(iv)[dvs].mean()
                
                interpretation_detail = ""
                if p_val < 0.05:
                    interpretation_detail = f"\n\n### Interpretation:\nThe groups defined by **{iv}** are significantly different when considering all outcomes together.\n"
                    interpretation_detail += "This suggests that the independent variable has an effect on the combination of dependent variables."
                else:
                    interpretation_detail = f"\n\n### Interpretation:\nNo significant difference was found between groups in **{iv}** when considering the outcomes together."
                
                summary_text += interpretation_detail

                for dv in dvs:
                    with PlotUtils.setup_plotting():
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.boxplot(x=iv, y=dv, data=df_clean, ax=ax, palette=palette)
                        ax.set_title(f"Distribution of {dv} by {iv}")
                        ax.set_xlabel(iv)
                        ax.set_ylabel(dv)
                        plt.tight_layout()
                        
                        artifacts.append({
                            "type": "plot",
                            "id": f"manova_boxplot_{dv}",
                            "title": f"Distribution: {dv}",
                            "content": PlotUtils.fig_to_base64(fig)
                        })
                        plt.close(fig)

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
        return results.get('summary', "MANOVA Completed.")