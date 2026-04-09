
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
            label="Categorical Variable (X-axis)", description="Optional variable to group columns. Leave blank for 1D plot.", required=False
        ))
        params.add_parameter(ToolParameter(
            name="numeric_variable", parameter_type=ParameterType.NUMERIC_COLUMN_SELECT,
            label="Numeric Variable (Y-axis)", description="Numeric variable for the plot.", required=True
        ))
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            # Use efficient loading from BaseAnalysisTool
            df = self.load_dataset()

            cat_var = kwargs.get('categorical_variable')
            num_var = kwargs.get('numeric_variable')

            # Identify target columns for batch processing if num_var is missing
            target_cols = []
            if num_var:
                target_cols = [num_var]
            else:
                # Fallback: All numeric columns
                target_cols = df.select_dtypes(include=['number']).columns.tolist()
                # Exclude categorical variable from being treated as a numeric target if it's in the list
                if cat_var in target_cols:
                    target_cols.remove(cat_var)

            if not target_cols:
                return {"status": "error", "error": "No numeric variables found for plotting.", "summary": "Missing numeric variables."}

            artifacts = []
            processed_cols = []

            for col in target_cols:
                if not cat_var:
                    # 1D Box Plot Stats for ECharts
                    series_data = df[col].dropna()
                    if series_data.empty or series_data.nunique() < 2:
                        artifacts.append({"type": "text", "title": f"Box Plot of {col}", "content": "Skipped: Not enough variance/unique values for a box plot."})
                        continue
                        
                    stats = [
                        float(series_data.min()),
                        float(series_data.quantile(0.25)),
                        float(series_data.median()),
                        float(series_data.quantile(0.75)),
                        float(series_data.max())
                    ]
                    chart_data = {
                        "type": "boxplot",
                        "title": f"Distribution of {col}",
                        "categories": [col],
                        "values": [stats],
                        "metadata": {"yAxisLabel": col}
                    }
                    
                    with PlotUtils.setup_plotting():
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.boxplot(y=series_data, ax=ax)
                        ax.set_title(f"Distribution of {col}")
                        plt.tight_layout()
                        artifacts.append(PlotUtils.to_artifact(fig, f"box_plot_1d_{col}", f"Box Plot of {col}", data_override=chart_data))
                        plt.close(fig)
                else:
                    # Comparison Box Plot (Grouped)
                    categories = sorted(df[cat_var].dropna().unique().tolist())
                    values = []
                    for cat in categories:
                        group = df[df[cat_var] == cat][col].dropna()
                        if not group.empty:
                            stats = [
                                float(group.min()),
                                float(group.quantile(0.25)),
                                float(group.median()),
                                float(group.quantile(0.75)),
                                float(group.max())
                            ]
                            values.append(stats)
                        else:
                            values.append([0, 0, 0, 0, 0])

                    chart_data = {
                        "type": "boxplot",
                        "title": f"Box Plot of {col} by {cat_var}",
                        "categories": categories,
                        "values": values,
                        "metadata": {"yAxisLabel": col}
                    }

                    if df[col].nunique() < 2:
                        artifacts.append({"type": "text", "title": f"Box Plot of {col} by {cat_var}", "content": "Skipped: Not enough numeric variance for a box plot."})
                        continue

                    with PlotUtils.setup_plotting():
                        fig, ax = plt.subplots(figsize=(12, 7))
                        sns.boxplot(data=df, x=cat_var, y=col, ax=ax)
                        ax.set_title(f"Box Plot of {col} by {cat_var}")
                        ax.tick_params(axis='x', rotation=45)
                        plt.tight_layout()
                        artifacts.append(PlotUtils.to_artifact(fig, f"box_plot_{col}", f"Box Plot of {col} by {cat_var}", data_override=chart_data))
                        plt.close(fig)
                
                processed_cols.append(col)

            summary = f"Generated {len(artifacts)} Box plot(s) for the following variables: {', '.join(processed_cols)}."
            if cat_var:
                summary += f" (Grouped by '{cat_var}')"

            return {
                "status": "ok", 
                "summary": summary, 
                "artifacts": artifacts, 
                "meta": {"tool_name": self.name, "parameters": kwargs, "processed_columns": processed_cols}
            }

        except Exception as e:
            self.log_error(e)
            return {"status": "error", "error": str(e), "summary": f"An unexpected error occurred: {str(e)}"}