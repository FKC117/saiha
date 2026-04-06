
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


class LineChartTool(BaseAnalysisTool):
    """
    A tool to generate a line chart, typically for time series data.
    """

    @property
    def name(self) -> str:
        return "line_chart"

    @property
    def description(self) -> str:
        return "Generates a line chart to visualize the trend of a numeric variable over time or another sequence."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name=self.name)
        params.add_parameter(ToolParameter(
            name="x_axis", parameter_type=ParameterType.DATE_COLUMN_SELECT,
            label="X-axis (Time/Sequence)", description="Select the date, time, or sequence column for the X-axis.", required=True
        ))
        params.add_parameter(ToolParameter(
            name="y_axis", parameter_type=ParameterType.NUMERIC_COLUMN_SELECT,
            label="Y-axis (Value)", description="Select the numeric variable to plot on the Y-axis.", required=True
        ))
        params.add_parameter(ToolParameter(
            name="hue", parameter_type=ParameterType.CATEGORICAL_COLUMN_SELECT,
            label="Group by (Optional)", description="Select a categorical variable to draw separate lines for each group.", required=False
        ))
        params.add_parameter(ToolParameter(
            name="moving_average_window", parameter_type=ParameterType.NUMBER,
            label="Moving Average Window (Optional)", description="Calculate and plot a moving average over this window size (e.g., 7 for a 7-day average).",
            required=False
        ))
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            # Use efficient loading from BaseAnalysisTool
            df = self.load_dataset()

            x_axis = kwargs.get('x_axis')
            y_axis = kwargs.get('y_axis')
            hue = kwargs.get('hue')
            moving_avg_window_val = kwargs.get('moving_average_window')
            moving_avg_window = None
            if moving_avg_window_val:
                try:
                    moving_avg_window = int(moving_avg_window_val)
                except (ValueError, TypeError):
                    moving_avg_window = None

            if not x_axis or not y_axis:
                return {"status": "error", "summary": "Both X-axis and Y-axis variables are required."}

            # Ensure the x-axis is sorted, especially for time series
            try:
                df[x_axis] = pd.to_datetime(df[x_axis])
            except (ValueError, TypeError):
                pass  # If it's not a date, just sort it as is

            # --- PERFORMANCE OPTIMIZATION ---
            # Pre-aggregate data before plotting to handle large datasets efficiently.
            # Instead of letting seaborn aggregate, we do it explicitly.
            group_by_cols = [x_axis]
            if hue and hue.strip():
                group_by_cols.append(hue)
            
            plot_df = df.groupby(group_by_cols)[y_axis].sum().reset_index()
            plot_df = plot_df.sort_values(by=x_axis)

            # Calculate moving average if requested
            if moving_avg_window:
                ma_col_name = f'{y_axis}_{moving_avg_window}_MA'
                if hue and hue.strip():
                    # Calculate moving average for each group
                    plot_df[ma_col_name] = plot_df.groupby(hue)[y_axis].transform(lambda x: x.rolling(window=moving_avg_window, min_periods=1).mean())
                else:
                    # Calculate moving average for the whole series
                    plot_df[ma_col_name] = plot_df[y_axis].rolling(window=moving_avg_window, min_periods=1).mean()

            summary = f"Line chart generated for '{y_axis}' over '{x_axis}'."
            artifacts: List[Dict[str, Any]] = []

            with PlotUtils.setup_plotting():
                fig, ax = plt.subplots(figsize=(14, 7))
                # Plot original line with some transparency
                sns.lineplot(data=plot_df, x=x_axis, y=y_axis, hue=hue if hue and hue.strip() else None, ax=ax, alpha=0.5, legend='full')
                if moving_avg_window:
                    # Plot moving average line
                    sns.lineplot(data=plot_df, x=x_axis, y=ma_col_name, hue=hue if hue and hue.strip() else None, ax=ax, linestyle='--', legend=False)
                    ax.set_title(f"Line Chart of {y_axis} with {moving_avg_window}-Period Moving Average")
                ax.set_title(f"Line Chart of {y_axis} over {x_axis}")
                artifacts.append({"type": "plot", "id": "line_chart", "title": f"Line Chart for {y_axis}", "content": PlotUtils.fig_to_base64(fig)})
                plt.close(fig)

            return {"status": "ok", "summary": summary, "artifacts": artifacts, "meta": {"tool_name": self.name, "parameters": kwargs}}

        except Exception as e:
            self.log_error(e)
            return {"status": "error", "summary": f"An unexpected error occurred: {str(e)}"}