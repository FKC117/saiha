"""
Compute Variable Tool
Creates a new variable by applying an arithmetic expression to existing variables.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType
from .plot_utils import PlotUtils

class ComputeVariableTool(BaseAnalysisTool):
    """Tool for computing new variables."""

    @property
    def name(self) -> str:
        return "compute_variable"

    @property
    def description(self) -> str:
        return "Create a new variable derived from existing columns (e.g., 'A + B')."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name="compute_variable")
        params.add_parameter(
            ToolParameter(
                name="new_column_name",
                parameter_type=ParameterType.TEXT,
                label="New Variable Name",
                description="Name of the new variable to create.",
                required=True
            )
        )
        params.add_parameter(
            ToolParameter(
                name="expression",
                parameter_type=ParameterType.TEXT,
                label="Expression",
                description="Arithmetic expression involving column names (e.g., 'ColA + ColB / 2').",
                required=True,
                help_text="Example: (column1 + column2) * 10"
            )
        )
        params.add_parameter(
            ToolParameter(
                name="save_as_new_dataset",
                parameter_type=ParameterType.CHECKBOX,
                label="Save as New Dataset",
                description="Create a new dataset with the new variable.",
                default_value=False,
                required=False
            )
        )
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            # 1. Get parameters and load data
            parameters = kwargs
            new_col_name = parameters.get("new_column_name")
            expression = parameters.get("expression")

            if not new_col_name or not expression:
                return {"status": "error", "summary": "Missing required parameters."}

            # We load the full dataset because the expression might reference any column.
            # In a real efficient implementation, we might parse the expression to find usage,
            # but loading all is safer for 'eval'.
            df = self.load_dataset()
            
            # Simple security check (very basic)
            # prevent import, exec, __, etc.
            if "__" in expression or "import" in expression or "exec" in expression:
                 return {"status": "error", "summary": "Invalid or unsafe expression detected."}

            # 2. Compute Variable
            try:
                # Using pandas eval for supported syntax
                # We interpret the expression in the context of the dataframe
                computed_series = df.eval(expression)
                
                # Check if result is a series (scalar result possible but unlikely for variable creation intent)
                if not isinstance(computed_series, (pd.Series, np.ndarray)):
                     # If scalar, expand to series
                     computed_series = pd.Series([computed_series] * len(df), index=df.index)
                     
                # Check if result is Timedelta (e.g., date differences)
                if pd.api.types.is_timedelta64_dtype(computed_series):
                     # Convert to days (float) for compatibility
                     computed_series = computed_series.dt.total_seconds() / 86400.0
            except Exception as e:
                return {"status": "error", "summary": f"Failed to evaluate expression: {str(e)}. Ensure column names are correct and contain no spaces without backticks."}

            if len(computed_series) != len(df):
                 return {"status": "error", "summary": "Computed result length mismatch."}

            # 3. Stats and Visualization
            artifacts = []
            sections = []

            # Summary metrics
            desc_stats = computed_series.describe()
            
            sections.append({
                'type': 'text',
                'title': 'Result Summary',
                'content': f"Successfully computed '{new_col_name}' using formula: '{expression}'."
            })
            
            metrics = ['count', 'mean', 'std', 'min', 'max']
            sections.append({
                'type': 'table',
                'title': 'Descriptive Statistics for New Variable',
                'headers': ['Metric', 'Value'],
                'data': [[m.capitalize(), f"{desc_stats.get(m, 0):.4f}"] for m in metrics]
            })
            
            # Preview with random sample
            preview_df = pd.DataFrame({new_col_name: computed_series})
            # Try to grab potential source columns for context? Hard to know which ones.
            # Just show the new column vs head.
            
            sections.append({
                'type': 'table',
                'title': 'First 10 Rows',
                'headers': [new_col_name],
                'data': [[f"{x:.4f}" if isinstance(x, (int, float)) else str(x)] for x in computed_series.head(10)]
            })


            if pd.api.types.is_numeric_dtype(computed_series):
                with PlotUtils.setup_plotting():
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.histplot(computed_series, kde=True, ax=ax, color='green')
                    ax.set_title(f"Distribution of New Variable: {new_col_name}")
                    
                    artifacts.append({
                        "type": "plot",
                        "id": "new_var_dist",
                        "title": "Distribution",
                        "content": PlotUtils.fig_to_base64(fig)
                    })
                    plt.close(fig)

            # Update DataFrame unconditionally
            df[new_col_name] = computed_series

            new_dataset_info = None
            save_as_new = parameters.get("save_as_new_dataset", False)
            if save_as_new:
                 from ...dataset_utils import save_dataframe_as_dataset
                 suffix = f"Computed ({new_col_name})"
                 new_dataset = save_dataframe_as_dataset(df, self.dataset, suffix)
                 
                 summary_msg = f"Created new variable '{new_col_name}'. Mean: {desc_stats.get('mean', 0):.2f}. Saved as new dataset: {new_dataset.name}"
                 new_dataset_info = {"id": str(new_dataset.id), "name": new_dataset.name}
            else:
                 self.save_dataset(df)
                 summary_msg = f"Created new variable '{new_col_name}'. Mean: {desc_stats.get('mean', 0):.2f}."

            return {
                "status": "ok",
                "summary": summary_msg,
                "sections": sections,
                "artifacts": artifacts,
                "meta": {
                    "tool_name": self.name,
                    "parameters": parameters,
                    "new_dataset": new_dataset_info
                }
            }

        except Exception as e:
            self.log_error(e)
            return {"status": "error", "summary": f"An unexpected error occurred: {str(e)}"}

    def interpret(self, results: Dict[str, Any]) -> Optional[str]:
        if results.get('status') != 'ok':
            return None
        return results.get('summary', "Computation Analysis Completed.")