
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

            # Hardened Param Fetching (Elite v3.3)
            # Uses 'variable' (schema) with fallbacks for legacy/hallucinated keys.
            variable = kwargs.get('variable') or kwargs.get('column') or kwargs.get('column_name')
            bins_str = kwargs.get('bins') or kwargs.get('num_bins')
            bins = 'auto'  # Default to automatic binning
            if bins_str:
                try:
                    bins = int(bins_str)
                except (ValueError, TypeError):
                    # If conversion fails, stick with 'auto'.
                    bins = 'auto'

            # Identify target columns for batch processing if variable is missing
            target_cols = []
            if variable:
              if variable in df.columns:
                target_cols = [variable]
              else:
                return {"status": "error", "error": f"Column '{variable}' not found in dataset.", "summary": f"Column '{variable}' not found in dataset."}
            else:
                # Fallback: All numeric columns
                target_cols = df.select_dtypes(include=['number']).columns.tolist()

            if not target_cols:
                return {"status": "error", "error": "No numeric variables found for plotting histograms.", "summary": "Missing numeric variables."}

            artifacts: List[Dict[str, Any]] = []
            processed_cols = []

            import numpy as np
            for col in target_cols:
                counts, bin_edges = np.histogram(df[col].dropna(), bins=bins if isinstance(bins, int) else 30)
                
                chart_data = {
                    "type": "bar",
                    "title": f"Distribution of {col}",
                    "xAxis": [f"{float(b):.2f}" for b in bin_edges[:-1]],
                    "series": [{"name": "Frequency", "data": [int(c) for c in counts]}],
                    "metadata": {"yAxisLabel": "Frequency", "xAxisLabel": col}
                }

                with PlotUtils.setup_plotting():
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(df[col], bins=bins, kde=True, ax=ax)
                    ax.set_title(f"Distribution of {col}")
                    ax.set_xlabel(col)
                    ax.set_ylabel("Frequency")
                    plt.tight_layout()
                    artifacts.append(PlotUtils.to_artifact(fig, f"histogram_{col}", f"Histogram of {col}", data_override=chart_data))
                    plt.close(fig)
                
                processed_cols.append(col)

            summary = f"Generated {len(artifacts)} Histogram(s) for variables: {', '.join(processed_cols)}."

            return {
                "status": "ok", 
                "summary": summary, 
                "artifacts": artifacts, 
                "meta": {"tool_name": self.name, "parameters": kwargs, "processed_columns": processed_cols}
            }

        except Exception as e:
            self.log_error(e)
            return {"status": "error", "error": str(e), "summary": f"An unexpected error occurred: {str(e)}"}