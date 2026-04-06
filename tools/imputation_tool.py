"""
Missing Value Imputation Tool
Fills missing values using various methods (Mean, Median, Mode, Constant).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
from sklearn.impute import SimpleImputer

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType
from .plot_utils import PlotUtils

class ImputationTool(BaseAnalysisTool):
    """Tool for imputing missing values."""
    is_destructive = True

    @property
    def name(self) -> str:
        return "imputation_tool"

    @property
    def description(self) -> str:
        return "Replace missing values with Mean, Median, Mode, or a Constant."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name="imputation_tool")
        params.add_parameter(
            ToolParameter(
                name="target_column",
                parameter_type=ParameterType.COLUMN_SELECT,
                label="Target Column",
                description="The column containing missing values to fill.",
                required=True
            )
        )
        params.add_parameter(
            ToolParameter(
                name="method",
                parameter_type=ParameterType.SELECT,
                label="Imputation Method",
                options=[
                    {"value": "mean", "label": "Mean (Average) - Numeric"},
                    {"value": "median", "label": "Median (Middle Value) - Numeric"},
                    {"value": "mode", "label": "Mode (Most Frequent) - All Types"},
                    {"value": "constant", "label": "Constant Value"},
                ],
                default_value="mean",
                required=True
            )
        )
        params.add_parameter(
            ToolParameter(
                name="constant_value",
                parameter_type=ParameterType.TEXT,
                label="Constant Value",
                description="Value to use if 'Constant' method is selected.",
                required=False
            )
        )
        params.add_parameter(
            ToolParameter(
                name="save_as_new_dataset",
                parameter_type=ParameterType.CHECKBOX,
                label="Save as New Dataset",
                description="Create a new dataset with imputed values.",
                default_value=False,
                required=False
            )
        )
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            # 1. Get parameters and load data
            parameters = kwargs
            target_col = parameters.get("target_column")
            method = parameters.get("method", "mean")
            constant_val = parameters.get("constant_value")

            if not target_col:
                return {"status": "error", "summary": "Please select a target column."}

            if not target_col:
                return {"status": "error", "summary": "Please select a target column."}

            save_as_new = parameters.get("save_as_new_dataset", False)
            if save_as_new:
                df = self.load_dataset()
            else:
                df = self.load_dataset()
                
            original_series = df[target_col].copy()
            
            missing_count = original_series.isna().sum()
            if missing_count == 0:
                return {"status": "ok", "summary": f"Column '{target_col}' has no missing values to impute.", "sections": [], "artifacts": []}

            imputed_series = None
            imputed_value = None

            # 2. Check types and apply imputation
            is_numeric = pd.api.types.is_numeric_dtype(original_series)

            if method in ['mean', 'median'] and not is_numeric:
                return {"status": "error", "summary": f"Method '{method}' requires a numeric column. '{target_col}' is {original_series.dtype}."}

            if method == "mean":
                imputer = SimpleImputer(strategy='mean')
                imputed_value = original_series.mean()
            elif method == "median":
                imputer = SimpleImputer(strategy='median')
                imputed_value = original_series.median()
            elif method == "mode":
                imputer = SimpleImputer(strategy='most_frequent')
                imputed_value = original_series.mode()[0]
            elif method == "constant":
                if constant_val is None:
                     return {"status": "error", "summary": "Please provide a Constant Value."}
                imputer = SimpleImputer(strategy='constant', fill_value=constant_val if not is_numeric else float(constant_val))
                imputed_value = constant_val

            # Reshape for sklearn
            reshaped_data = original_series.to_numpy().reshape(-1, 1)
            
            # Ensure compatibility with sklearn by replacing pd.NA with np.nan for object arrays
            if reshaped_data.dtype == object:
                 mask = pd.isna(reshaped_data)
                 reshaped_data[mask] = np.nan
            
            # Apply imputation
            if method == "constant" and is_numeric:
                 # Handle string/float mismatch for constant manually if needed, but sklearn does ok
                 pass

            imputed_data = imputer.fit_transform(reshaped_data)
            imputed_series = pd.Series(imputed_data.flatten(), index=original_series.index)


            artifacts = []
            sections = []

            # 3. Summary
            sections.append({
                'type': 'table',
                'title': 'Imputation Summary',
                'icon': 'fas fa-magic',
                'headers': ['Metric', 'Value'],
                'data': [
                   ['Original Missing Values', str(missing_count)],
                   ['Imputation Method', method.capitalize()],
                   ['Fill Value', str(imputed_value)]
                ]
            })

            # 4. Visualization (Distribution Comparison)
            if is_numeric:
                with PlotUtils.setup_plotting():
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Original (excluding NaN for plot)
                    sns.kdeplot(original_series.dropna(), label='Original (Before)', shade=True, color='blue', ax=ax)
                    
                    # Imputed
                    sns.kdeplot(imputed_series, label='Imputed (After)', shade=True, color='red', linestyle='--', ax=ax)
                    
                    ax.set_title(f"Distribution Before vs After Imputation ({target_col})")
                    ax.legend()
                    
                    artifacts.append({
                        "type": "plot",
                        "id": "imputation_dist",
                        "title": "Distribution Comparison",
                        "content": PlotUtils.fig_to_base64(fig)
                    })
                    plt.close(fig)

            new_dataset_info = None
            # Update DataFrame with imputed series unconditionally
            if imputed_series is not None:
                df[target_col] = imputed_series

            new_dataset_info = None
            if save_as_new:
                from ...dataset_utils import save_dataframe_as_dataset
                # Handle constant value in suffix if needed
                suffix_part = method.capitalize()
                if method == 'constant':
                     suffix_part = f"Constant ({constant_val})"
                     
                suffix = f"Imputed ({suffix_part})"
                new_dataset = save_dataframe_as_dataset(df, self.dataset, suffix)
                new_dataset_info = {"id": str(new_dataset.id), "name": new_dataset.name}
            else:
                self.save_dataset(df)

            return {
                "status": "ok",
                "summary": f"Imputed {missing_count} missing values in '{target_col}' using {method} (Value: {imputed_value})." + (f" Saved as: {new_dataset_info['name']}" if new_dataset_info else ""),
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
        return results.get('summary', "Imputation Analysis Completed.")