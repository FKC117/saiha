import abc
import logging
import time
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Type
from pydantic import BaseModel, ValidationError

# Dedicated Tool Logger (Configured in settings.py)
tool_logger = logging.getLogger('saiha.tools')

class ToolResult(BaseModel):
    """
    Standardized 'Viz-Ready' output for all tools.
    Matches frontend renderRichMessage expectations.
    """
    status: str  # 'success' | 'error'
    data: Dict[str, Any] = {} 
    artifacts: List[Dict[str, Any]] = [] 
    message: Optional[str] = None
    error: Optional[str] = None
    success: Optional[bool] = None # Legacy support

class BaseAnalysisTool(abc.ABC):
    """
    Universal Base Class for both Legacy and Hardened tools.
    Provides:
    1. Zero-Trust Pydantic Validation (Hardened Mode)
    2. Legacy Execute Bridge (Compatibility Mode)
    3. Optimized Data Loading (Parquet/Memory Cache)
    """
    name: str = "Base Tool"
    description: str = "Base analysis tool."
    input_schema: Optional[Type[BaseModel]] = None
    category: str = "General"

    def __init__(self, agent=None, **kwargs):
        self.agent = agent
        self.session = getattr(agent, 'session', None)
        self.dataset = getattr(agent, 'dataset', None)
        self.user = getattr(agent, 'user', None)
        self._created_dataset_info = None

    # --- LEGACY COMPATIBILITY SHIMS ---
    def log_error(self, e: Exception):
        """Legacy compatibility: shim for error logging."""
        tool_logger.error(f"Legacy error in {self.name}: {e}", exc_info=True)

    def log_execution(self, query: str, message: str, success: bool = True):
        """Legacy compatibility: shim for execution logging."""
        level = logging.INFO if success else logging.ERROR
        tool_logger.log(level, f"Legacy execution {self.name}: {message} (Query: {query})")

    def validate_dataset_requirement(self):
        """Legacy compatibility: ensures a dataset is present before execution."""
        if not self.dataset:
            raise ValueError(f"Tool '{self.name}' requires an active dataset.")
    # -----------------------------------

    def execute(self, query: str = "", **kwargs) -> Dict[str, Any]:
        """Legacy method to be implemented by ~80 tools."""
        raise NotImplementedError("Legacy tools must implement execute().")

    def run(self, df: pd.DataFrame, **kwargs) -> ToolResult:
        """New hardened method to be implemented by upgraded tools."""
        raise NotImplementedError("Hardened tools must implement run().")

    def validate_and_run(self, df: pd.DataFrame, params: Dict[str, Any]) -> ToolResult:
        """
        Unified entry point for the AnalysisAgent.
        Determines whether to run in 'Hardened' or 'Legacy' mode.
        """
        try:
            # 1. Parameter Validation (Strict Schema Layer)
            validated_params = params
            if self.input_schema:
                validated_params = self.input_schema(**params).dict()
            
            # 2. Execution Logic
            # Check if subclass has overridden 'run'
            if self.__class__.run != BaseAnalysisTool.run:
                return self.run(df, **validated_params)
            
            # Fallback to Legacy Bridge
            legacy_result = self.execute(query="", **validated_params)
            return self._normalize_legacy_result(legacy_result)

        except ValidationError as e:
            tool_logger.error(f"Validation failed for tool {self.name}: {e}")
            return ToolResult(status="error", error=str(e), message="Parameter validation failed.")
        except Exception as e:
            tool_logger.error(f"Error in tool {self.name}: {e}", exc_info=True)
            return ToolResult(status="error", error=str(e), message="Analysis failed.")

    def _normalize_legacy_result(self, result: Any) -> ToolResult:
        """Converts legacy dict or string outputs to standardized ToolResult."""
        if isinstance(result, str):
            # Handle tools that return direct Markdown/Text
            return ToolResult(
                status="success",
                message=result,
                data={},
                success=True
            )
            
        if not isinstance(result, dict):
            # Fallback for unexpected types
            return ToolResult(status="error", error=f"Unexpected tool output type: {type(result)}", message="Analysis failed.")

        status = "success" if result.get('status') in ['ok', 'success'] or result.get('success') else "error"
        error_msg = result.get('error')
        
        # Fallback for error message if status is error but error field is empty
        if status == "error" and not error_msg:
            error_msg = result.get('summary') or result.get('message') or "Analysis failed without specific error."

        # Bridge: Map legacy 'sections' and 'data' tables to modern 'artifacts'
        # This allows all 82 legacy tools to render tables/charts in the AI Chat.
        artifacts = result.get('artifacts', [])
        sections = result.get('sections', [])
        data = result.get('data', {})
        
        if not artifacts:
            # 1. From Sections
            if sections:
                for section in sections:
                    artifact_type = section.get('type')
                    if artifact_type in ['table', 'chart', 'plot', 'image']:
                        artifacts.append({
                            'type': artifact_type,
                            'label': section.get('title', 'Analysis Result'),
                            'headers': section.get('headers', []),
                            'data': section.get('data', []),
                            'metadata': section.get('metadata', {})
                        })
            
            # 2. From Data (Scan for tables like 'outlier_table')
            if isinstance(data, dict):
                for key, val in data.items():
                    if isinstance(val, dict) and 'records' in val:
                        # This looks like a legacy table in data dictionary
                        headers = val.get('headers')
                        records = val.get('records', [])
                        if not headers and records and isinstance(records[0], dict):
                            headers = list(records[0].keys())
                            data_rows = [list(r.values()) for r in records]
                        else:
                            data_rows = records
                        
                        artifacts.append({
                            'type': 'table',
                            'label': val.get('title', key.replace('_', ' ').title()),
                            'headers': headers or [],
                            'data': data_rows
                        })

        return ToolResult(
            status=status,
            data=data,
            artifacts=artifacts,
            message=result.get('summary', result.get('message')),
            error=error_msg,
            success=result.get('success')
        )

    def load_dataset(self, columns=None) -> pd.DataFrame:
        """
        Optimized dataset loader. Prefers cached _df from the executor (Pass-by-Memory).
        """
        # --- Memory Cache Optimization (Bug 12) ---
        if hasattr(self, '_df') and self._df is not None:
            return self._df
        
        from ..database_processing_logic.dataset_utils import load_dataset_data
        if not self.dataset:
            raise ValueError("Tool requires an active dataset.")
        return load_dataset_data(self.dataset.id, columns=columns)

    def save_dataset(self, df: pd.DataFrame, force_new: bool = False, new_name_suffix: str = "") -> dict:
        """
        Saves transformed data. Reused from legacy.
        """
        from ..database_processing_logic.dataset_utils import save_dataframe_as_dataset
        new_dataset = save_dataframe_as_dataset(df, self.dataset, new_name_suffix or f"Updated by {self.name}")
        self.dataset = new_dataset
        return {"id": str(new_dataset.id), "name": new_dataset.name}

    def clean_column_names(self, columns: List[str]) -> List[str]:
        """
        Sanitizes a list of column names for use in 'statsmodels' formulas.
        Mirror's DatasetProcessor's cleaning logic to ensure consistency.
        Example: 'Air Pollution (%)' -> 'Air_Pollution'
        """
        import re
        cleaned_cols = []
        for col in columns:
            if not col:
                cleaned_cols.append("unnamed_column")
                continue
            
            # 1. Replace non-alphanumeric with '_'
            sanitized = re.sub(r'[^0-9a-zA-Z_]', '_', str(col))
            # 2. Handle starting digits
            if sanitized and sanitized[0].isdigit():
                sanitized = '_' + sanitized
            # 3. Collapse multiple underscores and strip them from ends
            sanitized = re.sub(r'__+', '_', sanitized).strip('_')
            
            cleaned_cols.append(sanitized or "clean_column")
        
        return cleaned_cols
