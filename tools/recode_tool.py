"""
Recode Variable Tool
Maps existing values in a column to new values based on user-defined rules.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType
from .plot_utils import PlotUtils

class RecodeTool(BaseAnalysisTool):
    """Tool for recoding variable values."""

    @property
    def name(self) -> str:
        return "recode_tool"

    @property
    def description(self) -> str:
        return "Map existing values to new values (e.g., '1' to 'Male')."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name="recode_tool")
        params.add_parameter(
            ToolParameter(
                name="target_column",
                parameter_type=ParameterType.COLUMN_SELECT,
                label="Target Column",
                description="The column to recode.",
                required=True
            )
        )
        params.add_parameter(
            ToolParameter(
                name="mapping_rules",
                parameter_type=ParameterType.TEXTAREA,
                label="Mapping Rules",
                description="Enter rules in format 'OldValue:NewValue', separated by commas or newlines. (e.g., 1:Male, 2:Female)",
                required=True,
                help_text="Format: OldValue:NewValue (e.g., 1:Male, 2:Female)"
            )
        )
        params.add_parameter(
            ToolParameter(
                name="save_as_new_dataset",
                parameter_type=ParameterType.CHECKBOX,
                label="Save as New Dataset",
                description="Create a new dataset with the recoded variable.",
                default_value=False,
                required=False
            )
        )
        return params

    def parse_mapping_rules(self, rules_str: str) -> Dict[str, str]:
        """Parses the mapping rules string into a dictionary."""
        mapping = {}
        # improved splitting by both newline and comma
        import re
        parts = re.split(r'[,\n]', rules_str)
        
        for part in parts:
            if ':' in part:
                key, val = part.split(':', 1)
                key = key.strip()
                val = val.strip()
                if key:
                    mapping[key] = val
        return mapping

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            # 1. Get parameters
            parameters = kwargs
            target_col = parameters.get("target_column")
            rules_str = parameters.get("mapping_rules")

            if not target_col or not rules_str:
                return {"status": "error", "summary": "Missing required parameters."}

                return {"status": "error", "summary": "Missing required parameters."}

            save_as_new = parameters.get("save_as_new_dataset", False)
            if save_as_new:
                df = self.load_dataset()
            else:
                df = self.load_dataset()
                
            original_series = df[target_col].copy()

            # 2. Parse rules
            mapping = self.parse_mapping_rules(rules_str)
            if not mapping:
                return {"status": "error", "summary": "No valid mapping rules found. Please use 'Old:New' format."}

            # 3. Apply Recoding
            # We need to handle type matching. Keys in mapping are strings.
            # If original series is int, we might need to cast to str for matching, or cast keys to int.
            # Safer to cast series to string for the mapping process if it contains potential mixed types,
            # or try to cast keys to match valid types.
            
            # Simple approach: Convert series to string, map, then infer type
            temp_series = original_series.astype(str)
            recoded_series = temp_series.map(mapping).fillna(temp_series) # Keep original if not mapped
            
            # If the user intended to map to numbers, we can try to convert back
            # or just leave as object/string if it's categorical (likely)

            # 4. Visualization & Stats
            artifacts = []
            sections = []

            # Frequency Table Comparison
            freq_orig = original_series.value_counts().head(10)
            freq_new = recoded_series.value_counts().head(10)
            
            sections.append({
                'type': 'table',
                'title': 'Top 10 Values (Before vs After)',
                'headers': ['Value', 'Count (Original)', 'Value (Recoded)', 'Count (Recoded)'],
                'data': [] # Populated below or separate tables
            })
            # It's hard to align them row-by-row if they are different.
            # Let's show separate tables or just a summary.
            
            sections.pop() # Remove the complex table attempt
            
            sections.append({
                'type': 'table',
                'title': 'Original Values (Top 10)',
                'headers': ['Value', 'Count'],
                'data': [[str(k), str(v)] for k, v in freq_orig.items()]
            })

            sections.append({
                'type': 'table',
                'title': 'Recoded Values (Top 10)',
                'headers': ['Value', 'Count'],
                'data': [[str(k), str(v)] for k, v in freq_new.items()]
            })
            
            # Visual Comparison
            with PlotUtils.setup_plotting():
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                # Plot top categories
                freq_orig.plot(kind='bar', ax=axes[0], color='gray')
                axes[0].set_title('Original Distribution')
                
                freq_new.plot(kind='bar', ax=axes[1], color='purple')
                axes[1].set_title('Recoded Distribution')
                
                plt.tight_layout()
                artifacts.append({
                    "type": "plot",
                    "id": "recode_comparison",
                    "title": "Distribution Comparison",
                    "content": PlotUtils.fig_to_base64(fig)
                })
                plt.close(fig)

            # Update DataFrame unconditionally
            if recoded_series is not None:
                df[target_col] = recoded_series

            new_dataset_info = None
            if save_as_new and recoded_series is not None:
                from ...dataset_utils import save_dataframe_as_dataset
                suffix = f"Recoded ({target_col})"
                new_dataset = save_dataframe_as_dataset(df, self.dataset, suffix)
                summary = f"Recoded '{target_col}' using rules: {mapping}. Saved as new dataset: {new_dataset.name}"
                new_dataset_info = {"id": str(new_dataset.id), "name": new_dataset.name}
            else:
                self.save_dataset(df)
                summary = f"Recoded '{target_col}' using rules: {mapping}. Unique values changed from {original_series.nunique()} to {recoded_series.nunique()}."

            return {
                "status": "ok",
                "summary": summary,
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
        return results.get('summary', "Recode Analysis Completed.")