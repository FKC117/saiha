import pandas as pd
import numpy as np
from typing import Any, Dict

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from .plot_utils import PlotUtils

class DateTimeConversionTool(BaseAnalysisTool):
    """
    A tool to convert a column to DateTime format, handling various formats and errors.
    """

    @property
    def name(self) -> str:
        return "datetime_conversion"

    @property
    def description(self) -> str:
        return "Convert a text or numeric column to Date/Time format."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name=self.name)
        params.add_parameter(ToolParameter(
            name="column_to_convert",
            parameter_type=ParameterType.COLUMN_SELECT,
            label="Column to Convert",
            description="Select the column containing date/time values.",
            required=True,
            column_source="all"
        ))
        params.add_parameter(ToolParameter(
            name="format_string",
            parameter_type=ParameterType.TEXT,
            label="Format String (Optional)",
            description="e.g., '%Y-%m-%d' or '%d/%m/%Y'. Leave empty to auto-detect.",
            required=False,
            help_text="Provide if auto-detection fails. See Python datetime format codes."
        ))
        params.add_parameter(ToolParameter(
            name="dayfirst",
            parameter_type=ParameterType.CHECKBOX,
            label="Day First (e.g. 31/01/2020)",
            description="Check this if your data uses DD/MM/YYYY format (common outside US).",
            required=False,
            default_value=False
        ))
        params.add_parameter(ToolParameter(
            name="yearfirst",
            parameter_type=ParameterType.CHECKBOX,
            label="Year First (e.g. 2020/01/31)",
            description="Check this if your data uses YYYY/MM/DD format.",
            required=False,
            default_value=False
        ))
        params.add_parameter(ToolParameter(
            name="error_handling",
            parameter_type=ParameterType.SELECT,
            label="Error Handling",
            description="How to handle invalid values that cannot be converted.",
            required=True,
            default_value="coerce",
            options=[
                {"value": "coerce", "label": "Coerce (Set to NaT/Missing)"},
                {"value": "ignore", "label": "Ignore (Keep original value)"},
                {"value": "raise", "label": "Raise Error (Stop execution)"}
            ]
        ))
        params.add_parameter(ToolParameter(
            name="save_as_new_dataset",
            parameter_type=ParameterType.CHECKBOX,
            label="Save as New Dataset",
            description="Create a new dataset with the changes instead of modifying the current one.",
            required=False,
            default_value=False
        ))
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        import logging
        logger = logging.getLogger(__name__)
        try:
            col = kwargs.get("column_to_convert")
            fmt = kwargs.get("format_string") or None
            dayfirst = kwargs.get("dayfirst", False)
            yearfirst = kwargs.get("yearfirst", False)
            errors = kwargs.get("error_handling", "coerce")

            if not col:
                return {"status": "error", "summary": "Column to Convert is required."}

            # Load FULL dataset to ensure we don't drop other columns when saving
            df = self.load_dataset()
            
            if col not in df.columns:
                logger.error(f"DateTimeConversionTool: Column '{col}' not found in dataset {self.dataset.id}.")
                return {"status": "error", "summary": f"Column '{col}' not found in the dataset."}

            original_count = len(df)
            original_na_count = df[col].isna().sum()
            
            # Capture original values for comparison (Top 200 for performance)
            original_values = df[col].head(200).copy()

            # Attempt Conversion
            # Note: infer_datetime_format is deprecated in newer pandas, 
            # so we rely on default behavior which is quite smart.
            try:
                converted_series = pd.to_datetime(
                    df[col],
                    format=fmt,
                    dayfirst=dayfirst,
                    yearfirst=yearfirst,
                    errors=errors
                )
            except Exception as e:
                # This catches 'raise' errors or other parsing crashes
                return {"status": "error", "summary": f"Conversion failed: {str(e)}. Try using 'Coerce' to skip bad values, or verify your parameters."}

            if errors == 'ignore' and converted_series.dtype == object:
                 # If ignore was used and it failed to convert anything (remained object), we should warn?
                 # Actually, if errors='ignore', pandas returns the original input if conversion fails.
                 # Check if it's actually datetime
                 if not pd.api.types.is_datetime64_any_dtype(converted_series):
                     return {
                         "status": "ok",
                         "summary": f"Conversion ignored because errors were found. Column '{col}' remains unchanged.",
                         "sections": []
                     }

            # Update DataFrame
            df[col] = converted_series
            
            # Statistics
            new_na_count = df[col].isna().sum()
            failed_count = new_na_count - original_na_count
            
            # Generate Visualization: Conversion Status Bar Chart
            artifacts = []
            try:
                with PlotUtils.setup_plotting():
                    status_counts = pd.Series({
                        'Successful': original_count - new_na_count,
                        'Failed/NaT': new_na_count
                    })
                    
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.barplot(x=status_counts.index, y=status_counts.values, ax=ax, palette=['#2ecc71', '#e74c3c'])
                    ax.set_title(f'Conversion Status for {col}')
                    ax.set_ylabel('Count')
                    
                    artifacts.append({
                        "type": "plot", 
                        "id": "conversion_status_chart", 
                        "title": "Conversion Status Overview",
                        "content": PlotUtils.fig_to_base64(fig)
                    })
                    plt.close(fig)
            except Exception as e:
                logger.error(f"Failed to generate conversion chart: {e}")

            # Generate Comparison Table Data
            # Logic: Show top 10 successes and top 10 failures
            sections = []
            try:
                comparison_data = []
                
                # Create temporary comparison dataframe
                comp_df = pd.DataFrame({
                    'Original': original_values,
                    'Converted': df[col].head(200)
                })
                
                # Identify successes and failures in the sample
                failures = comp_df[comp_df['Converted'].isna() & comp_df['Original'].notna()].head(10)
                successes = comp_df[comp_df['Converted'].notna()].head(10)
                
                for idx, row in successes.iterrows():
                    comparison_data.append([str(row['Original']), str(row['Converted']), "Success"])
                    
                for idx, row in failures.iterrows():
                    comparison_data.append([str(row['Original']), "NaT", "Failed/Invalid"])
                
                if comparison_data:
                   sections.append({
                        'type': 'table',
                        'title': 'Before vs After (Sample)',
                        'icon': 'bi bi-arrow-left-right',
                        'headers': ['Original Value', 'Converted Value', 'Status'],
                        'data': comparison_data
                    })
            except Exception as e:
                 logger.error(f"Failed to generate comparison table: {e}")

            summary = f"Successfully converted column '{col}' to DateTime."
            if failed_count > 0:
                summary += f" Warning: {failed_count} values could not be converted and were set to NaT (Not a Time)."

            # Check for Save As New
            save_as_new = kwargs.get("save_as_new_dataset", False)
            if isinstance(save_as_new, str):
                save_as_new = save_as_new.lower() == 'true'
            
            if save_as_new:
                 from ...dataset_utils import save_dataframe_as_dataset
                 suffix = "DateTime"
                 new_dataset = save_dataframe_as_dataset(df, self.dataset, suffix)
                 summary += f" Saved as new dataset: **{new_dataset.name}**"
                 
                 return {
                    "status": "ok",
                    "summary": summary,
                    "data": {
                        "failed_count": int(failed_count),
                        "total_rows": int(original_count),
                        "new_dataset_id": str(new_dataset.id)
                    },
                    "sections": [
                        {
                            "type": "text",
                            "title": "Operation Successful",
                            "icon": "bi bi-check-circle-fill",
                            "content": summary
                        }
                    ] + sections,
                    "artifacts": artifacts
                }
            else:
                # Save
                self.save_dataset(df)
                
                return {
                    "status": "ok",
                    "summary": summary,
                    "data": {
                        "failed_count": int(failed_count),
                        "total_rows": int(original_count)
                    },
                    "sections": [
                        {
                            "type": "text",
                            "title": "Operation Successful",
                            "icon": "bi bi-check-circle-fill",
                            "content": summary
                        }
                    ] + sections,
                    "artifacts": artifacts
                }

        except Exception as e:
            self.log_error(e)
            return {"status": "error", "summary": f"An unexpected error occurred: {str(e)}"}
