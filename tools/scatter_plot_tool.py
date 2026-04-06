
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


class ScatterPlotTool(BaseAnalysisTool):
    """
    A tool to generate a scatter plot to visualize the relationship between two numeric variables.
    """

    @property
    def name(self) -> str:
        return "scatter_plot"

    @property
    def description(self) -> str:
        return "Creates a scatter plot to visualize the relationship between two numeric variables, optionally colored by a third categorical variable."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name=self.name)
        params.add_parameter(ToolParameter(
            name="x_axis", parameter_type=ParameterType.NUMERIC_COLUMN_SELECT,
            label="X-axis Variable", description="Select the numeric variable for the X-axis.", required=True
        ))
        params.add_parameter(ToolParameter(
            name="y_axis", parameter_type=ParameterType.NUMERIC_COLUMN_SELECT,
            label="Y-axis Variable", description="Select the numeric variable for the Y-axis.", required=True
        ))
        params.add_parameter(ToolParameter(
            name="hue", parameter_type=ParameterType.CATEGORICAL_COLUMN_SELECT,
            label="Color by (Optional)", description="Select a categorical variable to color the points.", required=False
        ))
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            # Use efficient loading from BaseAnalysisTool
            df = self.load_dataset()

            x_axis = kwargs.get('x_axis')
            y_axis = kwargs.get('y_axis')
            hue = kwargs.get('hue')
            if not hue or str(hue).strip() == "":
                hue = None

            if not x_axis or not y_axis:
                return {"status": "error", "summary": "Both X-axis and Y-axis variables are required."}

            summary = f"Scatter plot generated for '{y_axis}' vs. '{x_axis}'."
            artifacts: List[Dict[str, Any]] = []

            with PlotUtils.setup_plotting():
                fig, ax = plt.subplots(figsize=(10, 7))
                sns.scatterplot(data=df, x=x_axis, y=y_axis, hue=hue, ax=ax, alpha=0.7, s=50)
                ax.set_title(f"Scatter Plot of {y_axis} vs. {x_axis}")
                artifacts.append({"type": "plot", "id": "scatter_plot", "title": f"Scatter Plot of {y_axis} vs. {x_axis}", "content": PlotUtils.fig_to_base64(fig)})
                plt.close(fig)

            return {"status": "ok", "summary": summary, "artifacts": artifacts, "meta": {"tool_name": self.name, "parameters": kwargs}}

        except Exception as e:
            self.log_error(e)
            return {"status": "error", "summary": f"An unexpected error occurred: {str(e)}"}