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
        return ToolResult(
            status=status,
            data=result.get('data', result.get('metrics', {})),
            artifacts=result.get('artifacts', []),
            message=result.get('summary', result.get('message')),
            error=result.get('error'),
            success=result.get('success')
        )

    def load_dataset(self, columns=None) -> pd.DataFrame:
        """
        Optimized dataset loader. Reused from legacy with Parquet support.
        """
        from ..database_processing_logic.storage_manager_parquet import DatasetStorageManager
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
