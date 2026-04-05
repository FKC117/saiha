"""
Dataset Overview Tool
Provides a high-level overview of a dataset.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from django.core.files.storage import default_storage
from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet

class DatasetOverviewTool(BaseAnalysisTool):
    """Tool for getting comprehensive dataset overview."""
    
    @property
    def name(self) -> str:
        return "dataset_overview"
    
    @property
    def description(self) -> str:
        return "Provides a high-level overview of the dataset, including shape, data types, and memory usage."
    
    def get_parameters_schema(self) -> ToolParameterSet:
        # This tool does not require any parameters from the user.
        return ToolParameterSet(tool_name=self.name)

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        """Execute dataset overview analysis."""
        try:
            self.validate_dataset_requirement()
            
            # Load dataset from storage
            with default_storage.open(self.dataset.processed_file_path, 'rb') as f:
                df = pd.read_parquet(f)

            # --- Basic Stats ---
            num_rows, num_cols = df.shape
            memory_usage = df.memory_usage(deep=True).sum()
            
            # Format memory usage for readability
            if memory_usage > 1e9:
                mem_str = f"{memory_usage / 1e9:.2f} GB"
            elif memory_usage > 1e6:
                mem_str = f"{memory_usage / 1e6:.2f} MB"
            elif memory_usage > 1e3:
                mem_str = f"{memory_usage / 1e3:.2f} KB"
            else:
                mem_str = f"{memory_usage} bytes"

            basic_stats_data = [
                ['Number of Rows', num_rows],
                ['Number of Columns', num_cols],
                ['Total Memory Usage', mem_str]
            ]

            # --- Column Type Counts ---
            dtype_counts = df.dtypes.value_counts().reset_index()
            dtype_counts.columns = ['Data Type', 'Count']
            dtype_counts['Data Type'] = dtype_counts['Data Type'].astype(str)

            # --- Head and Tail ---
            df_head = df.head(5)
            df_tail = df.tail(5)

            summary = f"Dataset overview complete. The dataset has {num_rows} rows and {num_cols} columns."
            
            sections: List[Dict[str, Any]] = [
                {
                    'type': 'table', 'title': 'Dataset Statistics',
                    'headers': ['Statistic', 'Value'],
                    'data': basic_stats_data
                },
                {
                    'type': 'table', 'title': 'Column Data Types',
                    'headers': dtype_counts.columns.tolist(),
                    'data': dtype_counts.to_numpy().tolist()
                },
                {
                    'type': 'table', 'title': 'First 5 Rows (Head)',
                    'headers': df_head.columns.tolist(),
                    'data': df_head.to_numpy().tolist()
                },
                {
                    'type': 'table', 'title': 'Last 5 Rows (Tail)',
                    'headers': df_tail.columns.tolist(),
                    'data': df_tail.to_numpy().tolist()
                }
            ]

            return {
                "status": "ok",
                "summary": summary,
                "sections": sections,
                "artifacts": [],
                "meta": {"tool_name": self.name, "parameters": kwargs},
            }
            
        except Exception as e:
            self.log_error(e)
            return {"status": "error", "summary": f"An unexpected error occurred: {str(e)}"}

    def interpret(self, results: Dict[str, Any]) -> Optional[str]:
        """
        Provides a simple, rule-based interpretation of the dataset overview results.
        """
        if results.get('status') != 'ok':
            return None

        try:
            sections = results.get('sections', [])
            
            # Find 'Dataset Statistics' section
            stats_section = next((s for s in sections if s.get('title') == 'Dataset Statistics'), None)
            num_rows = 'an unknown number of'
            num_cols = 'an unknown number of'
            if stats_section:
                for row in stats_section.get('data', []):
                    if row[0] == 'Number of Rows':
                        num_rows = f"{row[1]:,}"
                    elif row[0] == 'Number of Columns':
                        num_cols = row[1]

            # Find 'Column Data Types' section
            dtypes_section = next((s for s in sections if s.get('title') == 'Column Data Types'), None)
            dtype_summary_parts = []
            if dtypes_section:
                for row in dtypes_section.get('data', []):
                    dtype_summary_parts.append(f"{row[1]} '{row[0]}'")

            if dtype_summary_parts:
                dtype_summary = ", ".join(dtype_summary_parts)
                return f"The dataset contains {num_rows} rows and {num_cols} columns. The column types consist of: {dtype_summary}."
            else:
                return f"The dataset contains {num_rows} rows and {num_cols} columns. A detailed breakdown of column types was not available."
        except Exception as e:
            return f"Could not automatically interpret the dataset overview due to an error: {e}"