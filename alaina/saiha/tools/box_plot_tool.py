# d:/quantly/quanta/quantalytics/ai_agents/tools/box_plot_tool.py

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


class BoxPlotTool(BaseAnalysisTool):
    """
    A tool to generate a box plot to compare distributions across categories.
    """

    @property
    def name(self) -> str:
        return "box_plot"

    @property
    def description(self) -> str:
        return "Creates a box plot to compare the distribution of a numeric variable across different categories."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name=self.name)
        params.add_parameter(ToolParameter(
            name="categorical_variable", parameter_type=ParameterType.CATEGORICAL_COLUMN_SELECT,
            label="Categorical Variable (X-axis)", description="Select the categorical variable to group by.", required=True
        ))
        params.add_parameter(ToolParameter(
            name="numeric_variable", parameter_type=ParameterType.NUMERIC_COLUMN_SELECT,
            label="Numeric Variable (Y-axis)", description="Select the numeric variable whose distribution will be plotted.", required=True
        ))
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            # Use efficient loading from BaseAnalysisTool
            df = self.load_dataset()

            cat_var = kwargs.get('categorical_variable')
            num_var = kwargs.get('numeric_variable')

            if not cat_var or not num_var:
                return {"status": "error", "summary": "Both a categorical and a numeric variable are required."}

            summary = f"Box plot generated for '{num_var}' grouped by '{cat_var}'."
            artifacts: List[Dict[str, Any]] = []

            with PlotUtils.setup_plotting():
                fig, ax = plt.subplots(figsize=(12, 7))
                sns.boxplot(data=df, x=cat_var, y=num_var, ax=ax)
                ax.set_title(f"Box Plot of {num_var} by {cat_var}")
                ax.tick_params(axis='x', rotation=45)
                plt.tight_layout()
                artifacts.append({"type": "plot", "id": "box_plot", "title": f"Box Plot of {num_var} by {cat_var}", "content": PlotUtils.fig_to_base64(fig)})
                plt.close(fig)

            return {"status": "ok", "summary": summary, "artifacts": artifacts, "meta": {"tool_name": self.name, "parameters": kwargs}}

        except Exception as e:
            self.log_error(e)
            return {"status": "error", "summary": f"An unexpected error occurred: {str(e)}"}