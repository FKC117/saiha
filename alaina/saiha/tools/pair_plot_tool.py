# d:/quantly/quanta/quantalytics/ai_agents/tools/pair_plot_tool.py

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
from django.core.files.storage import default_storage
import io
import base64

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType
from .plot_utils import PlotUtils


class PairPlotTool(BaseAnalysisTool):
    """
    A tool to generate a pair plot (scatter plot matrix).
    """

    @property
    def name(self) -> str:
        return "pair_plot"

    @property
    def description(self) -> str:
        return "Creates a matrix of scatter plots to visualize pairwise relationships between multiple numeric variables."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name=self.name)
        params.add_parameter(ToolParameter(
            name="numeric_variables",
            parameter_type=ParameterType.MULTISELECT,
            label="Numeric Variables",
            description="Select 2 to 5 numeric variables to include in the pair plot.",
            required=True,
            column_source="numeric"
        ))
        params.add_parameter(ToolParameter(
            name="hue",
            parameter_type=ParameterType.CATEGORICAL_COLUMN_SELECT,
            label="Color by (Optional)",
            description="Select a categorical variable to color the points.",
            required=False
        ))
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            # Use efficient loading from BaseAnalysisTool
            df = self.load_dataset()

            num_vars = kwargs.get('numeric_variables', [])
            if isinstance(num_vars, str): num_vars = [num_vars]
            # Sanitize hue: ensure it's None if empty or not provided
            hue = kwargs.get('hue') or None

            if not num_vars or len(num_vars) < 2:
                return {"status": "error", "summary": "Please select at least two numeric variables for the pair plot."}
            if len(num_vars) > 5:
                return {"status": "error", "summary": "To ensure readability, please select no more than 5 variables."}

            summary = f"Pair plot generated for variables: {', '.join(num_vars)}."
            artifacts: List[Dict[str, Any]] = []
            plot_vars = num_vars + [hue] if hue and hue not in num_vars else num_vars

            with PlotUtils.setup_plotting():
                pair_grid = sns.pairplot(df[plot_vars], hue=hue, corner=True, diag_kind='kde')
                pair_grid.fig.suptitle("Pairwise Relationship Plot", y=1.02)
                artifacts.append({"type": "plot", "id": "pair_plot", "title": "Pair Plot", "content": PlotUtils.fig_to_base64(pair_grid.fig)})
                plt.close(pair_grid.fig)

            return {"status": "ok", "summary": summary, "artifacts": artifacts, "meta": {"tool_name": self.name, "parameters": kwargs}}

        except Exception as e:
            self.log_error(e)
            return {"status": "error", "summary": f"An unexpected error occurred: {str(e)}"}