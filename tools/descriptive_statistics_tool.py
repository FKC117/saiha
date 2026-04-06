"""
Descriptive Statistics Tool
Performs comprehensive descriptive statistics analysis on numeric columns, following modern architecture.
"""

import math
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from .plot_utils import PlotUtils
from django.core.files.storage import default_storage
from typing import Dict, Any, List, Optional

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType


class DescriptiveStatisticsTool(BaseAnalysisTool):
    """Tool for performing descriptive statistics analysis."""
    
    @property
    def name(self) -> str:
        return "descriptive_statistics"
    
    @property
    def description(self) -> str:
        return "Perform comprehensive descriptive statistics analysis on numeric columns including mean, median, standard deviation, and more"
    
    def get_parameters_schema(self) -> ToolParameterSet:
        """Defines the parameters for the descriptive statistics tool."""
        params = ToolParameterSet(tool_name="descriptive_statistics")
        params.add_parameter(
            ToolParameter(
                name="numeric_columns",
                parameter_type=ParameterType.MULTISELECT,
                label="Select Numeric Columns",
                description="Choose numeric columns for descriptive statistics",
                required=False,
                help_text="Select one or more numeric columns to analyze.",
                column_source="numeric"
            )
        )
        params.add_parameter(
            ToolParameter(
                name="categorical_columns",
                parameter_type=ParameterType.MULTISELECT,
                label="Select Categorical Columns (Optional)",
                description="Choose categorical columns for frequency counts",
                required=False,
                help_text="Select one or more categorical columns to see their value counts.",
                column_source="categorical"
            )
        )
        params.add_parameter(
            ToolParameter(
                name="include_percentiles",
                parameter_type=ParameterType.CHECKBOX,
                label="Include Percentiles",
                description="Include 25th, 50th, and 75th percentiles.",
                required=False,
                default_value=True
            )
        )
        params.add_parameter(
            ToolParameter(
                name="include_skewness_kurtosis",
                parameter_type=ParameterType.CHECKBOX,
                label="Include Skewness & Kurtosis",
                description="Include measures of distribution shape.",
                required=False,
                default_value=True
            )
        )
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        """Executes descriptive statistics and returns a modern Result Envelope."""
        try:
            # 1. Get parameters and load data
            parameters = kwargs
            numeric_columns = parameters.get("numeric_columns", [])
            if isinstance(numeric_columns, str):
                numeric_columns = [numeric_columns]

            categorical_columns = parameters.get("categorical_columns", [])
            if isinstance(categorical_columns, str):
                categorical_columns = [categorical_columns]

            if not numeric_columns and not categorical_columns:
                return {"status": "error", "summary": "No columns were selected for analysis."}

            # Use efficient column projection loading from BaseAnalysisTool
            selected_columns = numeric_columns + categorical_columns
            df = self.load_dataset(columns=selected_columns)

            # 2. Perform analysis
            stats_df = pd.DataFrame()
            if numeric_columns:
                stats_df = df[numeric_columns].describe(percentiles=[.25, .5, .75]).transpose()
                if parameters.get('include_skewness_kurtosis', True):
                    stats_df['skew'] = df[numeric_columns].skew()
                    stats_df['kurtosis'] = df[numeric_columns].kurtosis()
                stats_df['missing'] = df[numeric_columns].isnull().sum()

            # 3. Generate Visualizations (as Static Images)
            artifacts = []

            # Generate individual plots for each numeric column
            if numeric_columns:
                with PlotUtils.setup_plotting():
                    for col in numeric_columns:
                        # Distribution Plot (Histogram + KDE)
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.histplot(df[col], kde=True, ax=ax)
                        ax.set_title(f'Distribution of {col}')
                        ax.set_xlabel(col)
                        ax.set_ylabel('Frequency')
                        
                        artifacts.append({
                            "type": "plot", 
                            "id": f"dist_{col}", 
                            "title": f'Distribution of {col}',
                            "content": PlotUtils.fig_to_base64(fig)
                        })
                        plt.close(fig)

                        # Violin Plot
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.violinplot(y=df[col], ax=ax)
                        ax.set_title(f'Violin Plot of {col}')
                        ax.set_ylabel(col)
                        
                        artifacts.append({
                            "type": "plot", 
                            "id": f"violin_{col}", 
                            "title": f'Violin Plot of {col}',
                            "content": PlotUtils.fig_to_base64(fig)
                        })
                        plt.close(fig)

            # Generate individual plots for each categorical column
            if categorical_columns:
                with PlotUtils.setup_plotting():
                    for col in categorical_columns:
                        # Show top 15 categories for readability
                        counts = df[col].value_counts().nlargest(15)
                        
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.barplot(x=counts.values, y=counts.index, ax=ax)
                        ax.set_title(f'Value Counts for {col}')
                        ax.set_xlabel('Count')
                        ax.set_ylabel(col)
                        
                        artifacts.append({
                            "type": "plot", 
                            "id": f"count_{col}", 
                            "title": f'Value Counts for {col}',
                            "content": PlotUtils.fig_to_base64(fig)
                        })
                        plt.close(fig)

            # 4. Construct the modern Result Envelope
            summary = f"Descriptive statistics calculated for {len(numeric_columns)} column(s): {', '.join(numeric_columns)}."
            if categorical_columns:
                summary += f" Frequency counts generated for {len(categorical_columns)} categorical column(s)."
            
            sections = []
            if not stats_df.empty:
                # Round float columns for cleaner presentation
                for col in stats_df.columns:
                    if pd.api.types.is_float_dtype(stats_df[col]):
                        stats_df[col] = stats_df[col].round(4)

                # Convert stats to a JSON-serializable format
                stats_df.reset_index(inplace=True)
                stats_df.rename(columns={'index': 'Column'}, inplace=True)
                stats_records = stats_df.to_dict(orient='records')
                
                # Create the sections list directly
                sections.append({
                    'type': 'table',
                    'title': 'Descriptive Statistics',
                    'icon': 'bi bi-table',
                    'headers': list(stats_df.columns),
                    'data': [list(rec.values()) for rec in stats_records]
                })

            return {
                "status": "ok",
                "summary": summary,
                "sections": sections,
                "artifacts": artifacts,
                "meta": {
                    "tool_name": self.name,
                    "parameters": parameters,
                    "columns_analyzed": numeric_columns + categorical_columns
                }
            }

        except Exception as e:
            self.log_error(e)
            return {"status": "error", "summary": f"An unexpected error occurred: {str(e)}"}

    def interpret(self, results: Dict[str, Any]) -> Optional[str]:
        """
        Provides a simple, rule-based interpretation of the descriptive statistics results.
        """
        if results.get('status') != 'ok':
            return None

        try:
            sections = results.get('sections', [])
            stats_table = next((s for s in sections if s.get('title') == 'Descriptive Statistics'), None)

            if not stats_table or not stats_table.get('data'):
                return "Descriptive statistics data not found for interpretation."

            headers = stats_table.get('headers', [])
            records = [dict(zip(headers, row)) for row in stats_table.get('data', [])]
            
            findings = []
            for record in records:
                col_name = record.get('Column')
                if 'missing' in record and record['missing'] > 0:
                    findings.append(f"'{col_name}' has {int(record['missing'])} missing values.")
                
                if 'skew' in record and abs(record['skew']) > 1.0:
                    skew_dir = "positively (right) skewed" if record['skew'] > 1.0 else "negatively (left) skewed"
                    findings.append(f"'{col_name}' is highly {skew_dir} (skewness: {record['skew']:.2f}).")

                if 'kurtosis' in record and record['kurtosis'] > 3.0:
                    findings.append(f"'{col_name}' has high kurtosis ({record['kurtosis']:.2f}), suggesting heavy tails or potential outliers.")

            if not findings:
                return "The analyzed numeric columns appear to be reasonably symmetrical and complete, with no significant skew, kurtosis, or missing values detected."

            return "Key observations: " + " ".join(findings) + " Review the plots and table for more details."

        except Exception as e:
            return f"Could not automatically interpret the results due to an error: {e}"

# Force recompile