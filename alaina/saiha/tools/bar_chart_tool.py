# d:/quantly/quanta/quantalytics/ai_agents/tools/bar_chart_tool.py

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
from django.core.files.storage import default_storage

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType
from .plot_utils import PlotUtils


class BarChartTool(BaseAnalysisTool):
    """
    A tool to generate a bar chart for a single categorical variable.
    """

    @property
    def name(self) -> str:
        return "bar_chart"

    @property
    def description(self) -> str:
        return "Generates a bar chart to visualize the frequency of values in a categorical variable."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name=self.name)
        params.add_parameter(ToolParameter(
            name="variable", parameter_type=ParameterType.CATEGORICAL_COLUMN_SELECT,
            label="Categorical Variable", description="Select the categorical variable to plot.", required=True
        ))
        params.add_parameter(ToolParameter(
            name="top_n", parameter_type=ParameterType.NUMBER,
            label="Number of Categories to Show (Optional)", description="Limit the plot to the top N most frequent categories. Default is 25.",
            required=False, default_value=25
        ))
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            # Use efficient loading from BaseAnalysisTool
            df = self.load_dataset()

            variable = kwargs.get('variable')
            top_n_val = kwargs.get('top_n')
            # Accept either int or digit-string for top_n; default to 25
            if isinstance(top_n_val, int):
                top_n = top_n_val
            elif isinstance(top_n_val, str) and top_n_val.isdigit():
                top_n = int(top_n_val)
            else:
                top_n = 25

            if not variable:
                return {"status": "error", "summary": "A categorical variable to plot is required."}

            summary = f"Bar chart generated for the top {top_n} categories of '{variable}'."
            artifacts: List[Dict[str, Any]] = []

            with PlotUtils.setup_plotting():
                fig, ax = plt.subplots(figsize=(12, 7))
                plot_data = df[variable].value_counts().nlargest(top_n)
                sns.barplot(x=plot_data.index, y=plot_data.values, ax=ax, palette="viridis")
                ax.set_title(f"Top {top_n} Value Frequencies for {variable}")
                ax.set_ylabel("Count")
                ax.tick_params(axis='x', rotation=45)
                plt.tight_layout()
                artifacts.append({"type": "plot", "id": "bar_chart", "title": f"Bar Chart for {variable}", "content": PlotUtils.fig_to_base64(fig)})
                plt.close(fig)

            return {"status": "ok", "summary": summary, "artifacts": artifacts, "meta": {"tool_name": self.name, "parameters": kwargs}}

        except Exception as e:
            self.log_error(e)
            return {"status": "error", "summary": f"An unexpected error occurred: {str(e)}"}