
import pandas as pd
import numpy as np
from django.core.files.storage import default_storage
from typing import Any, Dict, List, Optional

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType


class DataQualityTool(BaseAnalysisTool):
    """
    A comprehensive tool to assess the quality of a dataset.
    """

    @property
    def name(self) -> str:
        return "data_quality_assessment"

    @property
    def description(self) -> str:
        return "Performs a comprehensive data quality assessment, checking for missing values, duplicates, outliers, and more."

    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name=self.name)
        params.add_parameter(ToolParameter(
            name="check_missing", parameter_type=ParameterType.CHECKBOX,
            label="Check Missing Values", required=False, default_value=True
        ))
        params.add_parameter(ToolParameter(
            name="check_duplicates", parameter_type=ParameterType.CHECKBOX,
            label="Check Duplicate Rows", required=False, default_value=True
        ))
        params.add_parameter(ToolParameter(
            name="check_outliers", parameter_type=ParameterType.CHECKBOX,
            label="Check for Outliers (Numeric Columns)", required=False, default_value=True
        ))
        params.add_parameter(ToolParameter(
            name="check_low_variance", parameter_type=ParameterType.CHECKBOX,
            label="Check for Low Variance Columns", required=False, default_value=True
        ))
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        try:
            parameters = kwargs
            check_missing = str(parameters.get("check_missing", "true")).lower() in ('true', 'on', '1')
            check_duplicates = str(parameters.get("check_duplicates", "true")).lower() in ('true', 'on', '1')
            check_outliers = str(parameters.get("check_outliers", "true")).lower() in ('true', 'on', '1')
            check_low_variance = str(parameters.get("check_low_variance", "true")).lower() in ('true', 'on', '1')

            # Use efficient loading from BaseAnalysisTool
            df = self.load_dataset()

            total_rows = len(df)
            sections: List[Dict[str, Any]] = []
            summary_points = []
            passed_checks = []

            # --- Missing Value Check ---
            if check_missing:
                missing_info = df.isnull().sum()
                missing_info = missing_info[missing_info > 0].sort_values(ascending=False)
                if not missing_info.empty:
                    missing_df = pd.DataFrame({
                        'Column': missing_info.index,
                        'Missing Count': missing_info.values,
                        'Missing Percentage': (missing_info.values / total_rows * 100).round(2)
                    })
                    sections.append({
                        'type': 'table', 'title': 'Missing Value Analysis',
                        'headers': missing_df.columns.tolist(),
                        'data': missing_df.to_numpy().tolist()
                    })
                    summary_points.append(f"{len(missing_df)} columns have missing values.")
                else:
                    summary_points.append("No missing values found.")
                    passed_checks.append(['Missing Value Analysis', 'OK / No Issues Found'])

            # --- Duplicate Row Check ---
            if check_duplicates:
                num_duplicates = df.duplicated().sum()
                if num_duplicates > 0:
                    # Get all occurrences of duplicate rows to show them grouped together
                    duplicate_rows = df[df.duplicated(keep=False)].sort_values(by=df.columns.tolist())
                    
                    # Limit to first 100 for display purposes to not overwhelm the UI
                    duplicate_rows_display = duplicate_rows.head(100)

                    sections.append({
                        'type': 'table', 
                        'title': 'Duplicate Row Analysis',
                        'headers': duplicate_rows_display.columns.tolist(),
                        'data': duplicate_rows_display.to_numpy().tolist(),
                        'footer': f"Found {num_duplicates} duplicate rows ({num_duplicates/total_rows:.2%}). Showing up to 100 instances. Consider removing them to avoid bias."
                    })

                    summary_points.append(f"{num_duplicates} duplicate rows were found.")
                else:
                    summary_points.append("No duplicate rows found.")
                    passed_checks.append(['Duplicate Row Analysis', 'OK / No Issues Found'])

            # --- Outlier Check (IQR method) ---
            if check_outliers:
                numeric_cols = df.select_dtypes(include=np.number).columns
                outlier_data = []
                for col in numeric_cols:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                    if not outliers.empty:
                        outlier_data.append([col, len(outliers), f"{len(outliers)/total_rows:.2%}"])
                
                if outlier_data:
                    sections.append({
                        'type': 'table', 'title': 'Outlier Detection (IQR Method)',
                        'headers': ['Column', 'Outlier Count', 'Percentage'],
                        'data': outlier_data
                    })
                    summary_points.append(f"Outliers detected in {len(outlier_data)} numeric columns.")
                else:
                    summary_points.append("No significant outliers detected in numeric columns.")
                    passed_checks.append(['Outlier Detection (IQR)', 'OK / No Issues Found'])

            # --- Low Variance Check ---
            if check_low_variance:
                low_variance_cols = []
                for col in df.columns:
                    if df[col].nunique() == 1:
                        low_variance_cols.append([col, df[col].iloc[0]])
                
                if low_variance_cols:
                    sections.append({
                        'type': 'table', 'title': 'Low Variance Columns (Single Value)',
                        'headers': ['Column', 'Constant Value'],
                        'data': low_variance_cols,
                        'footer': "These columns have only one unique value and may not be useful for analysis."
                    })
                    summary_points.append(f"{len(low_variance_cols)} columns have zero variance.")
                else:
                    summary_points.append("No low-variance columns found.")
                    passed_checks.append(['Low Variance Columns', 'OK / No Issues Found'])

            # --- Add a consolidated table for passed checks ---
            if passed_checks:
                sections.append({
                    'type': 'table', 'title': 'Passed Data Quality Checks',
                    'headers': ['Check Performed', 'Status'],
                    'data': passed_checks
                })
            # --- Final Summary ---
            if not summary_points:
                summary = "No data quality checks were selected."
            else:
                summary = "Data quality assessment complete. " + " ".join(summary_points)

            return {
                "status": "ok",
                "summary": summary,
                "sections": sections,
                "artifacts": [],
                "meta": {"tool_name": self.name, "parameters": parameters},
            }

        except Exception as e:
            self.log_error(e)
            return {"status": "error", "summary": f"An unexpected error occurred: {str(e)}"}

    def interpret(self, results: Dict[str, Any]) -> Optional[str]:
        """
        Provides a simple, rule-based interpretation of the data quality results.
        """
        if results.get('status') != 'ok':
            return None

        sections = results.get('sections', [])
        findings = []

        for section in sections:
            title = section.get('title', '')
            # Check if the section has data, indicating an issue was found
            if section.get('data'):
                if 'Missing Value Analysis' in title:
                    findings.append("missing values were found in one or more columns.")
                elif 'Duplicate Row Analysis' in title:
                    findings.append("duplicate rows were detected in the dataset.")
                elif 'Outlier Detection' in title:
                    findings.append("potential outliers were identified in numeric columns.")
                elif 'Low Variance Columns' in title:
                    findings.append("one or more columns with constant values (zero variance) were found.")

        if not findings:
            return "The data quality assessment passed all selected checks. No significant issues regarding missing values, duplicates, outliers, or low variance were detected."

        # Construct a summary sentence
        return "The assessment found that " + ", ".join(findings) + " It is recommended to review the detailed tables and consider appropriate data cleaning steps."