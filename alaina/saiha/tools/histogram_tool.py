# d:/quantly/quanta/quantalytics/ai_agents/tools/histogram_tool.py

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


class HistogramTool(BaseAnalysisTool):
    """
    A tool to generate a histogram for a single numeric variable.
    """

    @property
    def name(self) -> str:
        return "histogram"

    @property
    def description(self) -> str:
        return "Generates a histogram to visualize the distribution of a single numeric variable."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name=self.name)
        params.add_parameter(ToolParameter(
            name="variable", parameter_type=ParameterType.NUMERIC_COLUMN_SELECT,
            label="Variable to Plot", description="Select the numeric variable for the histogram.", required=True
        ))
        params.add_parameter(ToolParameter(
            name="bins", parameter_type=ParameterType.NUMBER,
            label="Number of Bins (Optional)", description="The number of bars in the histogram. Leave blank for automatic.",
            required=False, default_value=30
        ))
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            # Use efficient loading from BaseAnalysisTool
            df = self.load_dataset()

            variable = kwargs.get('variable')
            bins_str = kwargs.get('bins')
            bins = 'auto'  # Default to automatic binning
            if bins_str:
                try:
                    bins = int(bins_str)
                except (ValueError, TypeError):
                    # If conversion fails, stick with 'auto'.
                    # This handles cases where bins might be None or an empty string.
                    bins = 'auto'

            if not variable:
                return {"status": "error", "summary": "A variable to plot is required."}

            summary = f"Histogram generated for the variable '{variable}'."
            artifacts: List[Dict[str, Any]] = []

            with PlotUtils.setup_plotting():
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(df[variable], bins=bins, kde=True, ax=ax)
                ax.set_title(f"Distribution of {variable}")
                ax.set_xlabel(variable)
                ax.set_ylabel("Frequency")
                artifacts.append({"type": "plot", "id": "histogram", "title": f"Histogram of {variable}", "content": PlotUtils.fig_to_base64(fig)})
                plt.close(fig)

            return {"status": "ok", "summary": summary, "artifacts": artifacts, "meta": {"tool_name": self.name, "parameters": kwargs}}

        except Exception as e:
            self.log_error(e)
            return {"status": "error", "summary": f"An unexpected error occurred: {str(e)}"}