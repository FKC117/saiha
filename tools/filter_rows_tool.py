"""
Filter Rows Tool
Filters the dataset based on specified conditions.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType
from .plot_utils import PlotUtils

class FilterRowsTool(BaseAnalysisTool):
    """Tool for filtering rows in a dataset."""
    is_destructive = True

    @property
    def name(self) -> str:
        return "filter_rows"

    @property
    def description(self) -> str:
        return "Filter the dataset based on a specific condition."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name="filter_rows")
        params.add_parameter(
            ToolParameter(
                name="filter_column",
                parameter_type=ParameterType.COLUMN_SELECT,
                label="Filter Column",
                description="The column to apply the filter on.",
                required=True
            )
        )
        params.add_parameter(
            ToolParameter(
                name="condition",
                parameter_type=ParameterType.SELECT,
                label="Condition",
                options=[
                    {"value": "equals", "label": "Equals (==)"},
                    {"value": "not_equals", "label": "Not Equals (!=)"},
                    {"value": "greater_than", "label": "Greater Than (>)"},
                    {"value": "less_than", "label": "Less Than (<)"},
                    {"value": "greater_equal", "label": "Greater or Equal (>=)"},
                    {"value": "less_equal", "label": "Less or Equal (<=)"},
                    {"value": "contains", "label": "Contains (Text)"},
                    {"value": "not_contains", "label": "Does Not Contain (Text)"},
                ],
                default_value="equals",
                required=True
            )
        )
        params.add_parameter(
            ToolParameter(
                name="value",
                parameter_type=ParameterType.TEXT,
                label="Value",
                description="The value to compare against.",
                required=True
            )
        )
        params.add_parameter(
            ToolParameter(
                name="save_as_new_dataset",
                parameter_type=ParameterType.CHECKBOX,
                label="Save as New Dataset",
                description="Create a new dataset with the filtered rows.",
                default_value=False,
                required=False
            )
        )
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            # 1. Get parameters and load data
            parameters = kwargs
            filter_col = parameters.get("filter_column")
            condition = parameters.get("condition")
            value_str = parameters.get("value")

            if not filter_col or not condition or value_str is None:
                return {"status": "error", "summary": "Missing required parameters."}

            df = self.load_dataset() # Load full dataset to show effect on whole data
            
            initial_count = len(df)
            
            # 2. Convert value to appropriate type
            col_dtype = df[filter_col].dtype
            comparison_value = value_str
            
            if pd.api.types.is_numeric_dtype(col_dtype):
                try:
                    comparison_value = float(value_str)
                except ValueError:
                    return {"status": "error", "summary": f"Column '{filter_col}' is numeric, but value '{value_str}' is not a valid number."}

            # 3. Apply Filter
            filtered_df = df.copy()
            
            if condition == "equals":
                filtered_df = df[df[filter_col] == comparison_value]
            elif condition == "not_equals":
                filtered_df = df[df[filter_col] != comparison_value]
            elif condition == "greater_than":
                filtered_df = df[df[filter_col] > comparison_value]
            elif condition == "less_than":
                filtered_df = df[df[filter_col] < comparison_value]
            elif condition == "greater_equal":
                filtered_df = df[df[filter_col] >= comparison_value]
            elif condition == "less_equal":
                filtered_df = df[df[filter_col] <= comparison_value]
            elif condition == "contains":
                filtered_df = df[df[filter_col].astype(str).str.contains(str(comparison_value), na=False)]
            elif condition == "not_contains":
                filtered_df = df[~df[filter_col].astype(str).str.contains(str(comparison_value), na=False)]
            
            final_count = len(filtered_df)
            removed_count = initial_count - final_count

            artifacts = []
            sections = []

            # 4. Summary Table
            sections.append({
                'type': 'table',
                'title': 'Filtering Summary',
                'icon': 'fas fa-filter',
                'headers': ['Metric', 'Count'],
                'data': [
                    ['Original Rows', str(initial_count)],
                    ['Rows After Filter', str(final_count)],
                    ['Rows Removed', str(removed_count)]
                ]
            })

            # 5. Visual Summary (Bar Chart of counts)
            with PlotUtils.setup_plotting():
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(['Original', 'Filtered'], [initial_count, final_count], color=['gray', 'blue'])
                ax.set_title('Row Count Comparison')
                ax.set_ylabel('Number of Rows')
                
                # Add labels on bars
                for i, v in enumerate([initial_count, final_count]):
                    ax.text(i, v, str(v), ha='center', va='bottom')
                
                plt.tight_layout()
                artifacts.append({
                    "type": "plot",
                    "id": "filter_counts",
                    "title": "Row Comparison",
                    "content": PlotUtils.fig_to_base64(fig)
                })
                plt.close(fig)

            # 6. Preview Table
            sections.append({
                'type': 'table',
                'title': f'First 10 Rows of Filtered Data',
                'headers': filtered_df.columns.tolist()[:10], # Limit cols for display if too wide
                'data': filtered_df.head(10).values.tolist()
            })

            summary = f"Filtered '{filter_col}' where {condition} '{value_str}'. Kept {final_count} rows ({removed_count} removed)."
            
            new_dataset_info = None
            save_as_new = parameters.get("save_as_new_dataset", False)
            
            if save_as_new:
                from ...dataset_utils import save_dataframe_as_dataset
                suffix = f"Filtered ({filter_col} {condition} {value_str})"
                # Truncate suffix if too long
                if len(suffix) > 50:
                    suffix = suffix[:47] + "..."
                    
                new_dataset = save_dataframe_as_dataset(filtered_df, self.dataset, suffix)
                summary += f" Saved as new dataset: {new_dataset.name}"
                new_dataset_info = {"id": str(new_dataset.id), "name": new_dataset.name}
            else:
                self.save_dataset(filtered_df)
                summary += " Dataset updated."

            return {
                "status": "ok",
                "summary": summary,
                "sections": sections,
                "artifacts": artifacts,
                "meta": {
                    "tool_name": self.name,
                    "parameters": parameters,
                    "final_row_count": final_count,
                    "new_dataset": new_dataset_info
                }
            }

        except Exception as e:
            self.log_error(e)
            return {"status": "error", "summary": f"An unexpected error occurred: {str(e)}"}

    def interpret(self, results: Dict[str, Any]) -> Optional[str]:
        if results.get('status') != 'ok':
            return None
        return results.get('summary', "Filter Analysis Completed.")