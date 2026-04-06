"""
Q-Q Plot Tool
Visualizes the quantiles of a variable against a theoretical distribution to check for normality.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats
from typing import Dict, Any, List, Optional

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType
from .plot_utils import PlotUtils

class QQPlotTool(BaseAnalysisTool):
    """Tool for generating Q-Q Plots."""

    @property
    def name(self) -> str:
        return "qq_plot"

    @property
    def description(self) -> str:
        return "Check if data follows a normal distribution using a Q-Q Plot."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name="qq_plot")
        params.add_parameter(
            ToolParameter(
                name="variable",
                parameter_type=ParameterType.NUMERIC_COLUMN_SELECT,
                label="Variable",
                description="Numeric variable to test.",
                required=True
            )
        )
        params.add_parameter(
            ToolParameter(
                name="distribution",
                parameter_type=ParameterType.SELECT,
                label="Theoretical Distribution",
                options=[
                    {"value": "norm", "label": "Normal"},
                    {"value": "t", "label": "Student's t"},
                    {"value": "uniform", "label": "Uniform"}
                ],
                default_value="norm",
                required=False
            )
        )
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            # 1. Get parameters
            parameters = kwargs
            col = parameters.get("variable")
            dist = parameters.get("distribution", "norm")

            if not col:
                return {"status": "error", "summary": "Please select a variable."}

            df = self.load_dataset(columns=[col])
            df_clean = df.dropna()

            if df_clean.empty:
                 return {"status": "error", "summary": "No data remaining after removing missing values."}

            data = df_clean[col]
            shapiro_text = "Normality test not performed."

            # Shapiro-Wilk Test (if sample size < 5000, otherwise Kolmogorov-Smirnov is often preferred, but Shapiro is good standard)
            # Scipy warns if N > 5000.
            shapiro_stat, shapiro_p = scipy.stats.shapiro(data)
            shapiro_text = f"### Normality Test (Shapiro-Wilk):\nStatistic: {shapiro_stat:.4f}, P-value: {shapiro_p:.4f}.\n"
            if shapiro_p < 0.05:
                shapiro_text += "Result: **Data is NOT normally distributed** (p < 0.05)."
            else:
                shapiro_text += "Result: **Data follows a normal distribution** (p >= 0.05)."
            
            if len(data) > 5000:
                shapiro_text += " (Note: Sample size large, p-value might be overly sensitive)."

            # 2. Visualization
            artifacts = []
            with PlotUtils.setup_plotting():
                fig, ax = plt.subplots(figsize=(6, 6))
                
                # Q-Q Plot
                # line='45' draws 45-degree reference line
                # line='s' fits a line to the standardized data
                sm.qqplot(data, line='s', ax=ax, dist=getattr(scipy.stats, dist) if hasattr(scipy.stats, dist) else scipy.stats.norm)
                # Note: statsmodels qqplot uses scipy.stats distributions if passed, but default string 'norm' works implicitly for normal.
                # However, for 't' or others, passing the scipy object or just relying on kwargs might be needed.
                # Simplest way for 'norm' is default.
                
                # Let's just use simple logic for 'norm' vs others.
                from scipy import stats
                dist_func = stats.norm
                dist_args = ()
                
                if dist == 't':
                     dist_func = stats.t
                     dist_args = (10,) # Default df=10 for t, user doesn't control yet.
                elif dist == 'uniform':
                    dist_func = stats.uniform
                
                # We need to re-draw to be safe
                plt.close(fig)
                fig, ax = plt.subplots(figsize=(6, 6))
                
                sm.qqplot(data, dist=dist_func, distargs=dist_args, line='s', ax=ax)
                
                ax.set_title(f"Q-Q Plot: {col} vs {dist}")
                
                artifacts.append({
                    "type": "plot",
                    "id": "qq_plot",
                    "title": "Q-Q Plot",
                    "content": PlotUtils.fig_to_base64(fig)
                })
                plt.close(fig)

            return {
                "status": "ok",
                "summary": f"Generated Q-Q Plot for '{col}' against '{dist}' distribution.\n\n{shapiro_text}",
                "sections": [],
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
        return results.get('summary', "Q-Q Plot Completed.")