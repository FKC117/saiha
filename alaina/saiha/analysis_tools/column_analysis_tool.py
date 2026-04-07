"""
Column Analysis Tool
Analyzes specific columns including distributions, missing values, and data types.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
from django.core.files.storage import default_storage
from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType
from .plot_utils import PlotUtils
import re


class ColumnAnalysisTool(BaseAnalysisTool):
    """Tool for analyzing specific columns."""
    
    @property
    def name(self) -> str:
        return "column_analysis"
    
    @property
    def description(self) -> str:
        return "Profiles all column types (numeric AND categorical): missing values, unique counts, distributions, and value frequencies. Use for data exploration and mixed-type profiling. Do NOT use when user asks for skewness, kurtosis, or detailed numeric statistics — use descriptive_statistics instead."
    
    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name=self.name)
        params.add_parameter(ToolParameter(
            name="columns_to_analyze",
            parameter_type=ParameterType.MULTISELECT,
            label="Select Columns to Analyze",
            description="Choose one or more columns to get a detailed analysis.",
            required=True,
            column_source="all"
        ))
        params.add_parameter(ToolParameter(
            name="generate_plots",
            parameter_type=ParameterType.CHECKBOX,
            label="Generate Visualizations",
            description="Generate plots for each selected column (histograms for numeric, bar charts for categorical).",
            required=False,
            default_value=True
        ))
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        """Execute column analysis."""
        try:
            parameters = kwargs
            columns_to_analyze = parameters.get("columns_to_analyze", [])
            if isinstance(columns_to_analyze, str):
                columns_to_analyze = [columns_to_analyze]
            gen_plots = str(parameters.get("generate_plots", "true")).lower() in ('true', 'on', '1')

            if not columns_to_analyze:
                return {"status": "error", "summary": "Please select at least one column to analyze."}

            # Ensure we have a list, even if a single column is passed
            if isinstance(columns_to_analyze, str):
                columns_to_analyze = [columns_to_analyze]

            self.validate_dataset_requirement()
            
            # Use efficient 'Pass-by-Memory' loading (Bug 12)
            df = self.load_dataset()
            
            artifacts: List[Dict[str, Any]] = []
            for col in columns_to_analyze:
                if col in df.columns:
                    # Make a copy to avoid SettingWithCopyWarning
                    col_data = df[col].copy()

                    # Attempt to convert object columns that might be numeric
                    original_dtype = col_data.dtype
                    if pd.api.types.is_object_dtype(original_dtype):
                        col_data_numeric = pd.to_numeric(col_data, errors='coerce')
                        # If conversion is successful for a good portion, use the numeric version
                        if col_data.notna().sum() > 0 and col_data_numeric.notna().sum() / col_data.notna().sum() > 0.8:
                            col_data = col_data_numeric
                    
                    # Basic stats for all columns
                    basic_stats = {
                        'Data Type': str(col_data.dtype),
                        'Missing Values': int(col_data.isnull().sum()),
                        'Missing Percentage': f"{(col_data.isnull().sum() / len(col_data) * 100):.2f}%",
                        'Unique Values': int(col_data.nunique()),
                    }
                    artifacts.append({
                        'type': 'table',
                        'title': f"Basic Statistics for '{col}'",
                        'headers': ['Statistic', 'Value'],
                        'data': list(basic_stats.items())
                    })

                    # Type-specific analysis
                    if pd.api.types.is_numeric_dtype(col_data) and not pd.api.types.is_bool_dtype(col_data):
                        numeric_stats = col_data.describe().round(4).to_dict()
                        artifacts.append({
                            'type': 'table',
                            'title': f"Descriptive Statistics for '{col}'",
                            'headers': ['Statistic', 'Value'],
                            'data': list(numeric_stats.items())
                        })
                        if gen_plots:
                            try:
                                with PlotUtils.setup_plotting():
                                    # Histogram and KDE
                                    fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
                                    sns.histplot(col_data, kde=True, ax=ax_hist)
                                    ax_hist.set_title(f"Distribution of '{col}'")
                                    artifacts.append(PlotUtils.to_artifact(fig_hist, f"hist_{col}", f"Histogram for '{col}'"))
                                    plt.close(fig_hist)

                                    # Box Plot (With Mandatory Stats for ECharts)
                                    fig_box, ax_box = plt.subplots(figsize=(10, 4))
                                    sns.boxplot(x=col_data, ax=ax_box)
                                    ax_box.set_title(f"Box Plot of '{col}'")
                                    
                                    # Calculate 5-number summary for ECharts
                                    stats = [
                                        float(col_data.min()),
                                        float(col_data.quantile(0.25)),
                                        float(col_data.median()),
                                        float(col_data.quantile(0.75)),
                                        float(col_data.max())
                                    ]
                                    chart_data = {
                                        "type": "boxplot",
                                        "title": f"Box Plot of {col}",
                                        "categories": [col],
                                        "values": [stats],
                                        "metadata": {"xAxisLabel": col}
                                    }
                                    
                                    artifacts.append(PlotUtils.to_artifact(fig_box, f"box_{col}", f"Box Plot for '{col}'", data_override=chart_data))
                                    plt.close(fig_box)
                            except Exception as pe:
                                logger.error(f"Plot generation failed for '{col}': {pe}")

                    elif pd.api.types.is_categorical_dtype(col_data) or pd.api.types.is_object_dtype(col_data) or pd.api.types.is_bool_dtype(col_data):
                        freq_table = col_data.value_counts(normalize=True).mul(100).round(2)
                        freq_table = freq_table.reset_index()
                        freq_table.columns = ['Value', 'Frequency (%)']
                        
                        # Limit to top 20 for readability
                        footer = None
                        if len(freq_table) > 20:
                            freq_table = freq_table.head(20)
                            footer = "Showing top 20 most frequent values."

                        artifacts.append({
                            'type': 'table',
                            'title': f"Value Frequencies for '{col}'",
                            'headers': freq_table.columns.tolist(),
                            'data': freq_table.to_numpy().tolist(),
                            'footer': footer
                        })
                        if gen_plots:
                            try:
                                with PlotUtils.setup_plotting():
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    plot_data = col_data.value_counts().nlargest(20)
                                    sns.barplot(x=plot_data.index.astype(str), y=plot_data.values, ax=ax, palette="viridis")
                                    ax.set_title(f"Top 20 Value Frequencies for '{col}'")
                                    ax.set_ylabel("Count")
                                    ax.tick_params(axis='x', rotation=45)
                                    plt.tight_layout()
                                    plot_res = PlotUtils.fig_to_base64(fig)
                                    artifacts.append({
                                        "type": plot_res['fallback_type'], 
                                        "id": f"bar_{col}", 
                                        "title": f"Bar Chart for '{col}'", 
                                        "metadata": plot_res['structured_data']
                                    })
                                    plt.close(fig)
                            except Exception as pe:
                                logger.error(f"Bar plot failed for '{col}': {pe}")
                    
                    elif pd.api.types.is_datetime64_any_dtype(col_data):
                        date_stats = {
                            'Earliest Date': str(col_data.min()),
                            'Latest Date': str(col_data.max()),
                        }
                        artifacts.append({
                            'type': 'table', 
                            'title': f"Date Range for '{col}'",
                            'headers': ['Statistic', 'Value'], 
                            'data': list(date_stats.items())
                        })
                        if gen_plots:
                            try:
                                with PlotUtils.setup_plotting():
                                    fig, ax = plt.subplots(figsize=(12, 6))
                                    col_data.value_counts().sort_index().plot(ax=ax)
                                    ax.set_title(f"Entries over Time for '{col}'")
                                    plot_res = PlotUtils.fig_to_base64(fig)
                                    artifacts.append({
                                        "type": plot_res['fallback_type'], 
                                        "id": f"timeline_{col}", 
                                        "title": f"Timeline for '{col}'", 
                                        "metadata": plot_res['structured_data']
                                    })
                                    plt.close(fig)
                            except Exception as pe:
                                logger.error(f"Timeline failed for '{col}': {pe}")
            
            summary = f"Completed analysis for {len(columns_to_analyze)} column(s)."

            return {
                "status": "ok",
                "summary": summary,
                "artifacts": artifacts,
                "meta": {"tool_name": self.name, "parameters": parameters},
            }
            
        except Exception as e:
            self.log_error(e)
            return {"status": "error", "summary": f"An unexpected error occurred: {str(e)}"}

    def interpret(self, results: Dict[str, Any]) -> Optional[str]:
        """
        Provides a simple, rule-based interpretation of the column analysis results.
        """
        if results.get('status') != 'ok':
            return None

        try:
            sections = results.get('sections', [])
            if not sections:
                return "No analysis was performed."

            # Group stats by column name
            column_stats = {}
            for section in sections:
                title = section.get('title', '')
                match = re.search(r"for '(.+)'", title)
                if not match:
                    continue
                
                col_name = match.group(1)
                if col_name not in column_stats:
                    column_stats[col_name] = {}

                for row in section.get('data', []):
                    stat_name, stat_value = row[0], row[1]
                    column_stats[col_name][stat_name] = stat_value

            # Build interpretation for each column
            findings = []
            for col_name, stats in column_stats.items():
                dtype = stats.get('Data Type', 'unknown')
                missing_pct = float(str(stats.get('Missing Percentage', '0%')).replace('%', ''))
                
                summary_part = f"'{col_name}' ({dtype}) has {stats.get('Unique Values', 'N/A')} unique values"
                if missing_pct > 0:
                    summary_part += f" and {missing_pct:.2f}% missing data."
                else:
                    summary_part += " and no missing data."
                findings.append(summary_part)

            if not findings:
                return "Could not extract key metrics from the analysis results."

            return " ".join(findings)
        except Exception as e:
            return f"Could not automatically interpret the column analysis results due to an error: {e}"
