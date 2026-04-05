# d:/quantly/quanta/quantalytics/ai_agents/tools/outlier_detection_tool.py

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # set backend BEFORE importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from django.core.files.storage import default_storage
from typing import Dict, Any, List, Tuple, Optional

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType
from quantalytics.ai_agents.tools.plot_utils import PlotUtils


class OutlierDetectionTool(BaseAnalysisTool):
    """
    A tool to detect outliers in numeric columns of a dataset using
    either the Interquartile Range (IQR) or Z-score method.
    """

    @property
    def name(self) -> str:
        return "outlier_detection"

    @property
    def description(self) -> str:
        return "Detects outliers in numeric columns using IQR or Z-score methods."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name=self.name)
        params.add_parameter(
            ToolParameter(
                name="numeric_columns",
                parameter_type=ParameterType.MULTISELECT,
                label="Select Numeric Columns",
                description="Choose one or more numeric columns to check for outliers.",
                required=True,
                help_text="Select columns to analyze for outliers.",
                column_source="numeric",
            )
        )
        params.add_parameter(
            ToolParameter(
                name="method",
                parameter_type=ParameterType.SELECT,
                label="Detection Method",
                description="The method to use for outlier detection.",
                required=True,
                default_value="iqr",
                options=[
                    {"value": "iqr", "label": "IQR (Interquartile Range)"},
                    {"value": "zscore", "label": "Z-score"},
                ],
            )
        )
        params.add_parameter(
            ToolParameter(
                name="iqr_multiplier",
                parameter_type=ParameterType.NUMBER,
                label="IQR Multiplier",
                description="Multiplier for the IQR score to identify outliers.",
                required=False,
                default_value=1.5,
                validation_rules={"min": 0.1, "step": "0.1"},
                help_text="Acceptable range: > 0. Default is 1.5 (standard) or 3 (extreme)."
            )
        )
        params.add_parameter(
            ToolParameter(
                name="zscore_threshold",
                parameter_type=ParameterType.NUMBER,
                label="Z-score Threshold",
                description="Number of standard deviations from the mean to be considered an outlier.",
                required=False,
                default_value=3.0,
                validation_rules={"min": 1.0, "step": "0.1"},
                help_text="Acceptable range: >= 1.0. Common values are 2.5, 3, or 3.5."
            )
        )
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            parameters = kwargs
            columns = parameters.get("numeric_columns", [])
            if isinstance(columns, str):
                columns = [columns]
            method = str(parameters.get("method", "iqr")).lower()
            if method not in {"iqr", "zscore"}:
                method = "iqr"
            
            # Use user-provided values with robust defaults.
            iqr_multiplier = float(parameters.get("iqr_multiplier", 1.5))
            zscore_threshold = float(parameters.get("zscore_threshold", 3.0))

            if not columns:
                return {"status": "error", "summary": "Please select at least one numeric column."}

            # Use efficient column projection loading from BaseAnalysisTool
            df = self.load_dataset(columns=columns)

            # Fail fast if any requested column is missing (keeps API shape the same below)
            missing = [c for c in columns if c not in df.columns]
            if missing:
                return {"status": "error", "summary": f"Column(s) not found in dataset: {', '.join(missing)}."}

            outlier_summary_records: List[Dict[str, Any]] = []
            total_outliers_found = 0
            artifacts: List[Dict[str, Any]] = []

            for col in columns:
                # Coerce to numeric, drop NaNs; if nothing left, skip quietly (matches your previous behavior)
                work_col = pd.to_numeric(df[col], errors="coerce").dropna()
                if work_col.empty:
                    # still render an empty-looking boxplot so UI doesn't look inconsistent
                    with PlotUtils.setup_plotting():
                        fig, ax = plt.subplots(figsize=(8, 5))
                        ax.set_title(f"Box Plot for '{col}' (no numeric data)")
                        ax.set_xlabel(col)
                        artifacts.append({
                            "type": "plot",
                            "id": f"boxplot_{col}",
                            "title": f"Box Plot for '{col}'",
                            "content": PlotUtils.fig_to_base64(fig),
                        })
                    continue

                if method == "iqr":
                    q1 = work_col.quantile(0.25)
                    q3 = work_col.quantile(0.75)
                    iqr = q3 - q1
                    # Guard: if iqr == 0 (no spread), there are no outliers under IQR rule
                    if not np.isfinite(iqr) or iqr == 0:
                        col_outliers = work_col.iloc[0:0]  # empty
                    else:
                        lower_bound = q1 - iqr_multiplier * iqr
                        upper_bound = q3 + iqr_multiplier * iqr
                        col_outliers = work_col[(work_col < lower_bound) | (work_col > upper_bound)]
                else:
                    mean = work_col.mean()
                    std = work_col.std(ddof=1)  # sample std
                    if not np.isfinite(std) or std == 0:
                        col_outliers = work_col.iloc[0:0]  # empty
                    else:
                        z_scores = (work_col - mean) / std
                        z_scores = z_scores.replace([np.inf, -np.inf], np.nan).dropna()
                        col_outliers = work_col.loc[z_scores.abs() > zscore_threshold]

                outlier_count = int(col_outliers.size)
                total_outliers_found += outlier_count

                # Keep EXACT table schema/keys you already use in api_views:
                if outlier_count > 0:
                    percentage = (outlier_count / int(work_col.size)) * 100.0
                    outlier_summary_records.append({
                        "Column Name": col,
                        "Number of Outliers": outlier_count,
                        "Percentage of Outliers": f"{percentage:.2f}%"
                    })

                # Preserve seaborn boxplot look (unchanged artifact structure)
                with PlotUtils.setup_plotting():
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.boxplot(x=work_col, ax=ax)
                    ax.set_title(f"Box Plot for '{col}'")
                    artifacts.append({
                        "type": "plot",
                        "id": f"boxplot_{col}",
                        "title": f"Box Plot for '{col}'",
                        "content": PlotUtils.fig_to_base64(fig),
                    })

            summary = (
                f"Outlier detection complete. Found a total of {total_outliers_found} outliers "
                f"across {len(columns)} selected column(s) using the '{method.upper()}' method."
            )

            # >>> DO NOT CHANGE OUTPUT SHAPE (tables) <<<
            return {
                "status": "ok",
                "summary": summary,
                "data": {
                    "outlier_table": {
                        "title": "Outlier Summary by Column",
                        "records": outlier_summary_records
                    }
                },
                "artifacts": artifacts,
                "meta": {
                    "tool_name": self.name,
                    "parameters": parameters,
                },
            }

        except Exception as e:
            self.log_error(e)
            return {"status": "error", "summary": f"An unexpected error occurred: {str(e)}"}

    def interpret(self, results: Dict[str, Any]) -> Optional[str]:
        """
        Provides a simple, rule-based interpretation of the outlier detection results.
        """
        if results.get('status') != 'ok':
            return None

        try:
            outlier_records = results.get('data', {}).get('outlier_table', {}).get('records', [])
            method = results.get('meta', {}).get('parameters', {}).get('method', 'iqr').upper()

            if not outlier_records:
                return f"No significant outliers were detected in the selected columns using the {method} method. The data appears to be clean in this regard."

            outlier_columns = [rec.get('Column Name') for rec in outlier_records if rec.get('Column Name')]
            
            if not outlier_columns:
                return "Outlier analysis was performed, but column names could not be identified in the results."

            col_list_str = ", ".join(f"'{col}'" for col in outlier_columns)
            return f"Using the {method} method, potential outliers were identified in the following column(s): {col_list_str}. It is recommended to review the box plots and consider whether these data points are errors or genuine extreme values."

        except Exception as e:
            return f"Could not automatically interpret the outlier results due to an error: {e}"