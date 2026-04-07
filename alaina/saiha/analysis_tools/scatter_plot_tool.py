
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

            # Identify target pairs for batch processing if axes are missing
            target_pairs = []
            if x_axis and y_axis:
                target_pairs = [(x_axis, y_axis)]
            else:
                # Fallback: Batch process first few numeric pairs
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                if len(numeric_cols) < 2:
                    return {"status": "error", "summary": "At least two numeric columns are required for a scatter plot."}
                
                # Create pairs (A, B), (A, C), (B, C) etc. limit to 3 pairs
                import itertools
                target_pairs = list(itertools.combinations(numeric_cols, 2))[:3]

            artifacts: List[Dict[str, Any]] = []
            processed_pairs = []

            for x, y in target_pairs:
                if x not in df.columns or y not in df.columns:
                    continue

                # Prepare ECharts scatter data
                series = []
                if hue:
                    groups = df.groupby(hue)
                    for name, group in groups:
                        series.append({
                            "name": str(name),
                            "data": group[[x, y]].dropna().values.tolist()
                        })
                else:
                    series.append({
                        "name": f"{y} vs {x}",
                        "data": df[[x, y]].dropna().values.tolist()
                    })

                chart_data = {
                    "type": "scatter",
                    "title": f"Relationship: {y} vs {x}",
                    "series": series,
                    "metadata": {
                        "xAxisLabel": x,
                        "yAxisLabel": y
                    }
                }

                with PlotUtils.setup_plotting():
                    fig, ax = plt.subplots(figsize=(10, 7))
                    sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax, alpha=0.7, s=50)
                    ax.set_title(f"Scatter Plot of {y} vs. {x}")
                    plt.tight_layout()
                    artifacts.append(PlotUtils.to_artifact(fig, f"scatter_{x}_{y}", f"Scatter Plot of {y} vs. {x}", data_override=chart_data))
                    plt.close(fig)
                
                processed_pairs.append(f"{y} vs {x}")

            summary = f"Generated {len(artifacts)} Scatter plot(s) for pairs: {', '.join(processed_pairs)}."
            if hue:
                summary += f" (Colored by '{hue}')"

            return {
                "status": "ok", 
                "summary": summary, 
                "artifacts": artifacts, 
                "meta": {"tool_name": self.name, "parameters": kwargs, "processed_pairs": processed_pairs}
            }

        except Exception as e:
            self.log_error(e)
            return {"status": "error", "summary": f"An unexpected error occurred: {str(e)}"}