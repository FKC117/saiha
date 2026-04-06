"""
Variable Transformation Tool
Applies various statistical transformations to valid columns (e.g., Log, Z-Score, Min-Max).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Any, List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType
from .plot_utils import PlotUtils

logger = logging.getLogger(__name__)

class VariableTransformationTool(BaseAnalysisTool):
    """Tool for applying statistical transformations to variables."""
    is_destructive = True

    @property
    def name(self) -> str:
        return "variable_transformation"

    @property
    def description(self) -> str:
        return "Apply statistical transformations (Log, Z-Score, Min-Max, etc.) to one or more variables."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name="variable_transformation")
        params.add_parameter(
            ToolParameter(
                name="target_columns",
                parameter_type=ParameterType.MULTISELECT,
                label="Target Variables",
                description="Select one or more variables to transform.",
                column_source="numeric", # Mostly numeric transformations
                required=True
            )
        )
        params.add_parameter(
            ToolParameter(
                name="transformation_type",
                parameter_type=ParameterType.SELECT,
                label="Transformation Type",
                options=[
                    {"value": "log", "label": "Log Transformation (Natural Log)"},
                    {"value": "sqrt", "label": "Square Root Transformation"},
                    {"value": "zscore", "label": "Z-Score Standardization (StandardScaler)"},
                    {"value": "minmax", "label": "Min-Max Scaling (Normalization)"},
                    {"value": "dummy", "label": "Dummy Coding (One-Hot Encoding)"},
                ],
                default_value="log",
                required=True
            )
        )
        params.add_parameter(
            ToolParameter(
                name="save_as_new_dataset",
                parameter_type=ParameterType.CHECKBOX,
                label="Save as New Dataset",
                description="Create a new dataset with the transformed variable(s).",
                default_value=False,
                required=False
            )
        )
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            # 1. Get parameters and load data
            parameters = kwargs
            target_cols = parameters.get("target_columns")
            if not target_cols:
                # Fallback for old calls or single selections
                target_cols = parameters.get("target_column")
            
            if not target_cols:
                return {"status": "error", "summary": "Please select at least one target variable."}
                
            if isinstance(target_cols, str):
                target_cols = [target_cols]

            trans_type = parameters.get("transformation_type", "log")
            save_as_new = parameters.get("save_as_new_dataset", False)
            
            df = self.load_dataset() # Load full dataset for multi-column processing and saving
            
            transformed_series_dict = {}
            warnings = []
            
            for target_col in target_cols:
                if target_col not in df.columns:
                    warnings.append(f"Column '{target_col}' not found in dataset.")
                    continue
                    
                original_data = df[target_col].dropna()
                if original_data.empty:
                    warnings.append(f"Column '{target_col}' is empty.")
                    continue

                transformed_data = None
                transformed_col_name = f"{target_col}_{trans_type}"
                
                # 2. Apply Transformation
                if trans_type == "log":
                    if pd.api.types.is_numeric_dtype(original_data):
                        if (original_data <= 0).any():
                            warnings.append(f"'{target_col}': contains non-positive values. Log transformation requires positive values.")
                            item_positive = original_data[original_data > 0]
                            transformed_data = np.log(item_positive)
                        else:
                            transformed_data = np.log(original_data)
                    else:
                        warnings.append(f"'{target_col}': Log transformation requires a numeric column.")

                elif trans_type == "sqrt":
                    if pd.api.types.is_numeric_dtype(original_data):
                        if (original_data < 0).any():
                            warnings.append(f"'{target_col}': contains negative values. Square root undefined for negative numbers.")
                            item_nonneg = original_data[original_data >= 0]
                            transformed_data = np.sqrt(item_nonneg)
                        else:
                            transformed_data = np.sqrt(original_data)
                    else:
                        warnings.append(f"'{target_col}': Square Root transformation requires a numeric column.")

                elif trans_type == "zscore":
                    if pd.api.types.is_numeric_dtype(original_data):
                        scaler = StandardScaler()
                        data_reshaped = original_data.values.reshape(-1, 1)
                        transformed_vals = scaler.fit_transform(data_reshaped)
                        transformed_data = pd.Series(transformed_vals.flatten(), index=original_data.index)
                    else:
                        warnings.append(f"'{target_col}': Z-Score transformation requires a numeric column.")

                elif trans_type == "minmax":
                    if pd.api.types.is_numeric_dtype(original_data):
                        scaler = MinMaxScaler()
                        data_reshaped = original_data.values.reshape(-1, 1)
                        transformed_vals = scaler.fit_transform(data_reshaped)
                        transformed_data = pd.Series(transformed_vals.flatten(), index=original_data.index)
                    else:
                        warnings.append(f"'{target_col}': Min-Max scaling requires a numeric column.")
                
                elif trans_type == "dummy":
                    dummy_df = pd.get_dummies(original_data, prefix=target_col, drop_first=False)
                    transformed_data = dummy_df
                
                if transformed_data is not None:
                    transformed_series_dict[target_col] = {
                        'transformed': transformed_data,
                        'original': original_data,
                        'name': transformed_col_name
                    }
            
            if not transformed_series_dict:
                summary_text = f"No valid columns could be transformed. Warnings: {'; '.join(warnings)}"
                return {"status": "error", "summary": summary_text}

            artifacts = []
            sections = []
            
            # 3. Generate Analysis Artifacts
            for col, info in transformed_series_dict.items():
                transformed_data = info['transformed']
                original_data = info['original']
                
                if trans_type == "dummy":
                    sections.append({
                        'type': 'text',
                        'title': f'Dummy Coding: {col}',
                        'content': f"Variable '{col}' was converted into {transformed_data.shape[1]} binary columns: {', '.join(transformed_data.columns)}."
                    })
                    
                    # Show head of dummy df
                    head_data = transformed_data.head(10).reset_index()
                    sections.append({
                        'type': 'table',
                        'title': f'First 10 Rows of Dummy Variables: {col}',
                        'headers': head_data.columns.tolist(),
                        'data': head_data.values.tolist()
                    })
                else:
                    stats_orig = original_data.describe()
                    stats_trans = transformed_data.describe()
                    
                    comp_data = []
                    metrics = ['count', 'mean', 'std', 'min', 'max']
                    for m in metrics:
                        comp_data.append([m.capitalize(), f"{stats_orig.get(m, 0):.4f}", f"{stats_trans.get(m, 0):.4f}"])
                    
                    sections.append({
                        'type': 'table',
                        'title': f'Statistics Comparison: {col}',
                        'headers': ['Metric', 'Before (Original)', 'After (Transformed)'],
                        'data': comp_data
                    })

                    # Visual Comparison
                    with PlotUtils.setup_plotting():
                        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                        sns.histplot(original_data, kde=True, ax=axes[0], color='blue')
                        axes[0].set_title(f'Original: {col}')
                        sns.histplot(transformed_data, kde=True, ax=axes[1], color='green')
                        axes[1].set_title(f'Transformed ({trans_type})')
                        plt.tight_layout()
                        
                        artifacts.append({
                            "type": "plot",
                            "id": f"plot_{col}",
                            "title": f"Distribution Change: {col}",
                            "content": PlotUtils.fig_to_base64(fig)
                        })
                        plt.close(fig)

            summary_text = f"Successfully transformed {len(transformed_series_dict)} variable(s) using {trans_type}."
            if warnings:
                summary_text += f" Warnings: {'; '.join(warnings)}"

            new_dataset_info = None
            # Update dataframe unconditionally
            for col, info in transformed_series_dict.items():
                transformed_data = info['transformed']
                transformed_col_name = info['name']
                if trans_type == "dummy":
                    df = pd.concat([df, transformed_data], axis=1)
                else:
                    df[transformed_col_name] = transformed_data

            new_dataset_info = None
            if save_as_new:
                from ...dataset_utils import save_dataframe_as_dataset
                suffix = f"Transformed {trans_type.capitalize()}"
                new_dataset = save_dataframe_as_dataset(df, self.dataset, suffix)
                summary_text += f" Saved as new dataset: {new_dataset.name}"
                new_dataset_info = {"id": str(new_dataset.id), "name": new_dataset.name}
            else:
                # Overwrite existing dataset
                self.save_dataset(df)
                summary_text += f" Dataset updated with {len(transformed_series_dict)} transformed variables."

            return {
                "status": "ok",
                "summary": summary_text,
                "sections": sections,
                "artifacts": artifacts,
                "meta": {
                    "tool_name": self.name,
                    "parameters": parameters,
                    "targets": list(transformed_series_dict.keys()),
                    "new_dataset": new_dataset_info
                }
            }

        except Exception as e:
            self.log_error(e)
            return {"status": "error", "summary": f"An unexpected error occurred: {str(e)}"}

    def interpret(self, results: Dict[str, Any]) -> Optional[str]:
        if results.get('status') != 'ok':
            return None
        return results.get('summary', "Transformation Analysis Completed.")
