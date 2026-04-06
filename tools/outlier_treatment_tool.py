"""
Outlier Treatment Tool
Treats outliers in numeric columns using various methods:
Winsorization, Trimming, Capping, and Imputation.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
from scipy.stats import mstats

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType
from .plot_utils import PlotUtils

class OutlierTreatmentTool(BaseAnalysisTool):
    """Tool for treating outliers in datasets."""

    @property
    def name(self) -> str:
        return "outlier_treatment"

    @property
    def description(self) -> str:
        return "Treat outliers using Winsorization, Trimming, Capping, or Replacement."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name="outlier_treatment")
        
        params.add_parameter(
            ToolParameter(
                name="target_columns",
                parameter_type=ParameterType.MULTISELECT,  # Changed to MULTISELECT
                label="Target Variables",
                description="Select one or more numeric variables to treat.",
                required=True,
                column_source="numeric"
            )
        )
        
        params.add_parameter(
            ToolParameter(
                name="method",
                parameter_type=ParameterType.SELECT,
                label="Treatment Method",
                options=[
                    {"value": "winsorize", "label": "Winsorization (Cap at percentiles)"},
                    {"value": "trim", "label": "Trimming (Remove rows)"},
                    {"value": "iqr_cap", "label": "IQR Capping (Replace limits with IQR bounds)"},
                    {"value": "zscore_cap", "label": "Z-Score Capping (Replace limits with Z-score bounds)"},
                    {"value": "mean_replace", "label": "Replace Outliers with Mean"},
                    {"value": "median_replace", "label": "Replace Outliers with Median"},
                ],
                default_value="winsorize",
                required=True
            )
        )
        
        params.add_parameter(
            ToolParameter(
                name="min_percentile",
                parameter_type=ParameterType.NUMBER,
                label="Lower Percentile (0-0.5)",
                description="Percentile for lower bound (e.g., 0.05 for 5th percentile). Used for Winsorize/Trim.",
                default_value=0.05,
                required=False,
                validation_rules={"min": 0.0, "max": 0.5, "step": "0.01"}
            )
        )
        
        params.add_parameter(
            ToolParameter(
                name="max_percentile",
                parameter_type=ParameterType.NUMBER,
                label="Upper Percentile (0.5-1.0)",
                description="Percentile for upper bound (e.g., 0.95 for 95th percentile). Used for Winsorize/Trim.",
                default_value=0.95,
                required=False,
                validation_rules={"min": 0.5, "max": 1.0, "step": "0.01"}
            )
        )

        params.add_parameter(
            ToolParameter(
                name="save_as_new_dataset",
                parameter_type=ParameterType.CHECKBOX, # Changed to CHECKBOX
                label="Save as New Dataset",
                description="If checked, creates a new dataset with the treated values.",
                default_value=False,
                required=False
            )
        )

        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            # 1. Load Parameters
            parameters = kwargs
            target_cols = parameters.get("target_columns")
            
            # Handle string input (single column) just in case
            if isinstance(target_cols, str):
                target_cols = [target_cols]
                
            if not target_cols:
                return {"status": "error", "summary": "Please select at least one numeric column."}

            method = parameters.get("method", "winsorize")
            min_p = float(parameters.get("min_percentile", 0.05))
            max_p = float(parameters.get("max_percentile", 0.95))

            # Load data - if saving, we might need full dataset eventually, 
            # but for processing speed we start with target columns.
            # However, simpler approach for saving is to load everything if we intend to save.
            save_as_new = parameters.get("save_as_new_dataset", False)
            
            if save_as_new:
                 # Load full dataset so we can save it all back
                 df = self.load_dataset()
            else:
                 # Always load full dataset to support overwriting
                 df = self.load_dataset()
            
            summary_actions = []
            sections = []
            artifacts = []
            processed_count = 0
            
            for col in target_cols:
                original_series = pd.to_numeric(df[col], errors='coerce').dropna()
                
                if original_series.empty:
                    # Skip empty/non-numeric columns with a warning in production, 
                    # but for now we append a message to summary
                    raw_series = df[col]
                    if raw_series.isna().all():
                         summary_actions.append(f"Skipped '{col}': Column contains only missing values.")
                    else:
                         summary_actions.append(f"Skipped '{col}': Column does not contain numeric data.")
                    continue

                processed_count += 1
                treated_series = original_series.copy()
                col_action_msg = ""
                
                # 2. Apply Treatment
                if method == "winsorize":
                    lower_limit = min_p
                    upper_limit = 1.0 - max_p
                    treated_array = mstats.winsorize(original_series, limits=[lower_limit, upper_limit])
                    treated_series = pd.Series(treated_array, index=original_series.index)
                    
                    # Update the main dataframe
                    df[col] = treated_series
                    
                    col_action_msg = f"Winsorized ({min_p*100:.1f}th-{max_p*100:.1f}th)."

                elif method == "trim":
                    lower_bound = original_series.quantile(min_p)
                    upper_bound = original_series.quantile(max_p)
                    mask = (original_series >= lower_bound) & (original_series <= upper_bound)
                    treated_series = original_series[mask]
                    
                    # Trimming changes row count, so applied to the whole DF for consistency? 
                    # If we trim one col, we must likely trim the dataset rows or introduce NaNs.
                    # Usually trimming means removing the rows from analysis.
                    # For "Save as New", we should probably filter the whole DF.
                    df = df.loc[mask]
                    
                    removed = len(original_series) - len(treated_series)
                    col_action_msg = f"Trimmed {removed} rows."
                    
                elif method == "iqr_cap":
                    q1 = original_series.quantile(0.25)
                    q3 = original_series.quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    treated_series = original_series.clip(lower=lower_bound, upper=upper_bound)
                    
                    # Update DF
                    df[col] = treated_series
                    
                    col_action_msg = f"IQR Capped."

                elif method == "zscore_cap":
                    mean = original_series.mean()
                    mean = original_series.mean()
                    std = original_series.std()
                    lower_bound = mean - 3 * std
                    upper_bound = mean + 3 * std
                    treated_series = original_series.clip(lower=lower_bound, upper=upper_bound)
                    
                    # Update DF
                    df[col] = treated_series
                    
                    col_action_msg = f"Z-Score Capped."

                elif method in ["mean_replace", "median_replace"]:
                    q1 = original_series.quantile(0.25)
                    q3 = original_series.quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    outliers_mask = (original_series < lower) | (original_series > upper)
                    outlier_count = outliers_mask.sum()
                    if outlier_count > 0:
                        replacement = original_series.mean() if method == "mean_replace" else original_series.median()
                        treated_series.loc[outliers_mask] = replacement
                        
                        # Update DF
                        df[col] = treated_series
                        
                        col_action_msg = f"Replaced {outlier_count} outliers."
                    else:
                        col_action_msg = "No outliers found."

                summary_actions.append(f"{col}: {col_action_msg}")
                
                 # Compare stats for this column
                stats_orig = original_series.describe()
                stats_treated = treated_series.describe()
                
                comp_data = []
                for m in ['count', 'mean', 'std', 'min', 'max']:
                    comp_data.append([m.capitalize(), f"{stats_orig.get(m, 0):.4f}", f"{stats_treated.get(m, 0):.4f}"])
                
                sections.append({
                    'type': 'table',
                    'title': f'Impact Analysis: {col}',
                    'headers': ['Statistic', 'Before', 'After'],
                    'data': comp_data
                })

                # Visual Comparison
                with PlotUtils.setup_plotting():
                    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                    sns.boxplot(x=original_series, ax=axes[0], color='lightblue')
                    axes[0].set_title(f'Original ({col})')
                    sns.boxplot(x=treated_series, ax=axes[1], color='lightgreen')
                    axes[1].set_title(f'After {method}')
                    plt.tight_layout()
                    artifacts.append({
                        "type": "plot",
                        "id": f"treatment_comparison_{col}",
                        "title": f"Comparison: {col}",
                        "content": PlotUtils.fig_to_base64(fig)
                    })
                    plt.close(fig)

                if processed_count == 0:
                     return {"status": "error", "summary": "No valid numeric columns were processed. " + " ".join(summary_actions)}
            
                     return {"status": "error", "summary": "No valid numeric columns were processed. " + " ".join(summary_actions)}
            
            # Save as new dataset if requested
            # save_as_new was retrieved earlier
            new_dataset_info = None
            
            if save_as_new:
                from ...dataset_utils import save_dataframe_as_dataset
                # Create a specific suffix
                suffix = f"Outlier Treated ({method})"
                new_dataset = save_dataframe_as_dataset(df, self.dataset, suffix)
                
                summary_actions.append(f"Saved to new dataset: '{new_dataset.name}'")
                new_dataset_info = {
                    "id": str(new_dataset.id),
                    "name": new_dataset.name
                }
            else:
                self.save_dataset(df)
                summary_actions.append("Dataset updated.")

            return {
                "status": "ok",
                "summary": "Processed " + str(processed_count) + " columns. " + " | ".join(summary_actions),
                "sections": sections,
                "artifacts": artifacts,
                "meta": {
                    "tool_name": self.name,
                    "parameters": parameters,
                    "processed_columns": target_cols,
                    "new_dataset": new_dataset_info
                }
            }

        except Exception as e:
            self.log_error(e)
            return {"status": "error", "summary": f"Error treating outliers: {str(e)}"}

    def interpret(self, results: Dict[str, Any]) -> Optional[str]:
        if results.get('status') != 'ok':
            return None
        return results.get('summary', "Outlier Treatment Completed.")
