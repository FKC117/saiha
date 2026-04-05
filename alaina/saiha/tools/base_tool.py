from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import re
import time
import logging
import pandas as pd
from django.core.exceptions import ValidationError
from .tool_parameters import ToolParameterSet
from .parameter_validator import ParameterValidator, ValidationError as ParameterValidationError

# Performance helpers (opt-in, safe imports)
try:
    from quantalytics.performance import AdvancedCache, request_timer_context, collect_process_stats
except Exception:  # pragma: no cover - optional runtime deps
    AdvancedCache = None  # type: ignore
    request_timer_context = None  # type: ignore
    collect_process_stats = None  # type: ignore

logger = logging.getLogger(__name__)

# module-level cache to reduce repeated dataset loads across tools in the same process
_GLOBAL_CACHE = AdvancedCache() if AdvancedCache is not None else None

class BaseAnalysisTool(ABC):
    """
    Abstract base class for all analysis tools.
    Defines the common interface that all tools must implement.
    """
    name: str = "Base Tool"
    description: str = "This is a base tool and should not be used directly."
    tool_type: str = "base_tool"
    category: str = "Other"
    is_destructive: bool = False

    def __init__(self, agent=None, **kwargs):
        self.agent = agent
        self.session = getattr(agent, 'session', None)
        self.dataset = getattr(agent, 'dataset', None)
        self.user = getattr(agent, 'user', None)
        self._created_dataset_info = None

    @abstractmethod
    def execute(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool's analysis logic.
        Should be implemented by subclasses.
        """
        pass

    def run(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Executes the tool and ensures output strictly follows the standardized schema.
        Agents should call `tool.run()` instead of `tool.execute()`.
        """
        start_time = time.time()
        try:
            result = self.execute(query, **kwargs)
            execution_time = time.time() - start_time
            
            # If the tool already returned the standardized format, normalize missing keys
            if isinstance(result, dict) and 'success' in result and ('data' in result or 'metrics' in result):
                if 'metrics' not in result:
                    result['metrics'] = {}
                if 'data' not in result:
                    result['data'] = {}
                if 'error' not in result:
                    result['error'] = None
                if 'meta' not in result:
                    result['meta'] = {}
                
                result['meta']['tool_name'] = self.name
                result['meta']['execution_time'] = execution_time
                if self._created_dataset_info:
                    result['meta']['new_dataset'] = self._created_dataset_info
                return result
                
            # Otherwise package it into standardized format
            return {
                "success": True,
                "data": result if isinstance(result, dict) else {"result": result},
                "metrics": {},
                "error": None,
                "meta": {
                    "tool_name": self.name,
                    "execution_time": execution_time,
                    "new_dataset": self._created_dataset_info
                }
            }
        except Exception as e:
            execution_time = time.time() - start_time
            self.log_error(e)
            return {
                "success": False,
                "data": {},
                "metrics": {},
                "error": {
                    "error_type": type(e).__name__,
                    "message": str(e)
                },
                "meta": {
                    "tool_name": self.name,
                    "execution_time": execution_time
                }
            }

    def log_error(self, error: Exception) -> None:
        """
        Log an exception with traceback to the tool errors log.
        """
        logger.error(f"Error in tool '{self.name}': {error}", exc_info=True)

    def get_parameters_schema(self) -> ToolParameterSet:
        """
        Returns the parameter schema for the tool.
        Tools that do not require parameters can use this default implementation.
        """
        from .tool_parameters import parameter_registry
        # Try to get centrally registered parameters first
        registered_params = parameter_registry.get_parameters(self.name)
        if registered_params:
            return registered_params
        return ToolParameterSet(tool_name=self.name)

    def validate_parameters(
        self,
        parameters: Dict[str, Any],
        strict: bool = False,
        collect_all: bool = True
    ) -> Tuple[bool, List[ParameterValidationError], List]:
        """
        Validate tool parameters against the schema.
        
        This is an optional, non-breaking method. Existing tools continue to work
        without validation. New tools can opt-in by calling this method.
        
        Args:
            parameters: Dictionary of parameter name -> value
            strict: If True, warnings are treated as errors
            collect_all: If True, collect all errors before returning; otherwise fail-fast
        
        Returns:
            Tuple of (is_valid: bool, errors: List[ValidationError], warnings: List[ValidationWarning])
        
        Raises:
            ParameterValidationError: If strict=True and there are errors/warnings
        
        Example:
            is_valid, errors, warnings = tool.validate_parameters(params)
            if not is_valid:
                for error in errors:
                    logger.error(f"{error.field}: {error.message}")
        """
        schema = self.get_parameters_schema()
        validator = ParameterValidator(schema, dataset=self.dataset)
        
        is_valid, errors, warnings = validator.validate(
            parameters,
            strict=strict,
            collect_all=collect_all
        )
        
        if not is_valid and strict:
            if errors:
                raise errors[0]  # Raise first error in strict mode
        
        return is_valid, errors, warnings
    
    def safe_validate_parameters(
        self,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Safely validate parameters with automatic error handling.
        
        Returns a dict with keys:
        - 'valid': bool
        - 'errors': list of error dicts with 'field' and 'message'
        - 'warnings': list of warning dicts with 'field' and 'message'
        - 'parameters': the original parameters (even if invalid)
        
        This method never raises exceptions, making it safe for API/UI usage.
        
        Example:
            result = tool.safe_validate_parameters(params)
            if result['valid']:
                execute_tool(result['parameters'])
        """
        try:
            is_valid, errors, warnings = self.validate_parameters(parameters)
            return {
                'valid': is_valid,
                'errors': [
                    {'field': e.field, 'message': e.message, 'value': str(e.value)}
                    for e in errors
                ],
                'warnings': [
                    {'field': w.field, 'message': w.message, 'severity': w.severity}
                    for w in warnings
                ],
                'parameters': parameters
            }
        except Exception as e:
            logger.exception("Error during parameter validation")
            return {
                'valid': False,
                'errors': [{'field': 'unknown', 'message': str(e)}],
                'warnings': [],
                'parameters': parameters
            }

    def validate_dataset_requirement(self):
        """
        Validates that a dataset is available if the tool requires it.
        """
        if not self.dataset:
            raise ValueError(f"The '{self.name}' tool requires an active dataset. Please select one first.")

    def clean_column_names(self, columns: list) -> list:
        """
        Sanitizes a list of column names to be valid Python identifiers for use in formulas.
        """
        cleaned_columns = []
        for col in columns:
            # Replace invalid characters with underscores
            sanitized_col = re.sub(r'[^0-9a-zA-Z_]', '_', str(col))
            # Remove leading numbers if they exist
            if sanitized_col and sanitized_col[0].isdigit():
                sanitized_col = '_' + sanitized_col
            # Replace multiple underscores with a single one and strip ends
            sanitized_col = re.sub(r'__+', '_', sanitized_col).strip('_')
            cleaned_columns.append(sanitized_col)
        return cleaned_columns
    
    def load_dataset(self, columns=None) -> pd.DataFrame:
        """
        Load dataset with optional column projection for memory efficiency.
        This method uses Parquet column projection when available.
        
        Args:
            columns: Optional list of column names to load. If None, load all columns.
                    Use this for memory efficiency when only specific columns are needed.
        
        Returns:
            pandas.DataFrame: The loaded dataset with specified columns
        
        Raises:
            ValueError: If no dataset is set for this tool
        """
        from ...dataset_utils import load_dataset_data

        self.validate_dataset_requirement()

        cache_key = None
        if _GLOBAL_CACHE is not None:
            try:
                cols_key = ",".join(columns) if columns else "__ALL__"
                cache_key = f"load_dataset:{self.dataset.id}:{cols_key}"
                cached = _GLOBAL_CACHE.get(cache_key)
                if cached is not None:
                    logger.debug(f"Dataset load cache hit: dataset={self.dataset.id} columns={cols_key}")
                    return cached
            except Exception:
                logger.exception("Error accessing global cache; proceeding without cache")

        # Time the load and collect lightweight process stats
        start = time.time()
        if request_timer_context is not None:
            # use generic name; operator can enable Prometheus if desired
            ctx = request_timer_context("dataset.load")
        else:
            ctx = None

        try:
            if ctx is not None:
                ctx.__enter__()
            df = load_dataset_data(self.dataset.id, columns=columns)
        finally:
            if ctx is not None:
                try:
                    ctx.__exit__(None, None, None)
                except Exception:
                    logger.exception("request_timer_context exit failed")

        duration = time.time() - start

        # Collect process-level stats if available
        proc_stats = {}
        try:
            if collect_process_stats is not None:
                proc_stats = collect_process_stats()
        except Exception:
            logger.exception("collect_process_stats failed")

        try:
            num_columns = len(df.columns) if hasattr(df, 'columns') else 0
            num_rows = len(df) if hasattr(df, '__len__') else 0
        except Exception:
            num_columns = 0
            num_rows = 0

        logger.info(
            "Tool '%s' loaded dataset %s (rows=%s cols=%s) in %.3fs; proc_stats=%s",
            self.name, getattr(self.dataset, 'id', '<unknown>'), num_rows, num_columns, duration, proc_stats,
        )

        # Store in cache for quick reuse (best-effort)
        if cache_key and _GLOBAL_CACHE is not None:
            try:
                # TTL of 5 minutes by default; operator can adjust via AdvancedCache init
                _GLOBAL_CACHE.set(cache_key, df, ttl=300)
            except Exception:
                logger.exception("Failed to store dataset in cache; continuing")

        return df

    def save_dataset(self, df: pd.DataFrame, force_new: bool = False, new_name_suffix: str = "") -> dict:
        """
        Overwrite the current dataset OR create a new version if the tool is destructive.
        Updates self.dataset to the new object safely so subsequent operations don't fail.
        
        Args:
            df: The DataFrame to save
            force_new: Explicitly override to save as a new dataset
            new_name_suffix: Suffix appended to dataset name if saving as new
            
        Returns:
            dict containing dataset info: {"id": str, "name": str, "is_new": bool}
        """
        import os
        from django.conf import settings
        from ...models import DatasetColumn
        from ...dataset_processor import DatasetProcessor
        
        self.validate_dataset_requirement()
        
        # 0. Immutability Layer: Spin up child dataset if destructive
        if self.is_destructive or force_new:
            from ...dataset_utils import save_dataframe_as_dataset
            suffix = new_name_suffix or f"Transformed by {self.name}"
            try:
                new_dataset = save_dataframe_as_dataset(df, self.dataset, suffix)
                logger.info(f"Is_destructive enabled ({self.name}). Created child dataset: {new_dataset.id}")
                self.dataset = new_dataset
                self._created_dataset_info = {"id": str(new_dataset.id), "name": new_dataset.name, "is_new": True}
                return self._created_dataset_info
            except Exception as e:
                logger.error(f"Immutability Layer failed to fork dataset for {self.name}: {e}")
                raise e

        
        logger.error(f"DEBUG: save_dataset received df with columns: {list(df.columns)}")
        
        if not self.dataset.processed_file_path:
             raise ValueError("Dataset has no processed file path.")
             
        full_path = os.path.join(settings.MEDIA_ROOT, self.dataset.processed_file_path)
        
        # 0. Invalidate Cache IMMEDIATELY
        if _GLOBAL_CACHE is not None:
             try:
                 # Invalidate all cached loads for this dataset ID
                 _GLOBAL_CACHE.delete_pattern(f"load_dataset:{self.dataset.id}:*")
             except Exception:
                 logger.exception("Failed to invalidate cache during save_dataset")

        # 1. Save File (Parquet preferred for performance)
        # We respect the existing format if possible, but default to parquet for processed data
        if self.dataset.storage_format == 'parquet' or full_path.endswith('.parquet'):
            df.to_parquet(full_path, index=False)
        else:
            df.to_csv(full_path, index=False)
            
        # 2. Update Dataset Metadata
        self.dataset.rows_count = len(df)
        self.dataset.columns_count = len(df.columns)
        self.dataset.file_size = os.path.getsize(full_path)
        self.dataset.save()
        
        # 3. Regenerate Column Metadata
        # We must delete old columns and recreate to match the new schema (renames, drops, type changes)
        processor = DatasetProcessor()
        columns_metadata = processor.get_column_metadata(df)
        
        # Transactional update recommended but simple delete-create works for now
        self.dataset.columns.all().delete()
        
        new_columns = []
        for col_meta in columns_metadata:
            new_columns.append(DatasetColumn(
                dataset=self.dataset,
                column_name=col_meta['column_name'],
                column_index=col_meta['column_index'],
                data_type=col_meta['data_type'],
                null_count=col_meta['null_count'],
                unique_count=col_meta['unique_count']
            ))
        
        DatasetColumn.objects.bulk_create(new_columns)
        
        # 4. Update JSON Metadata & Preview (Critical for UI consistency)
        try:
            from ...storage_manager_parquet import DatasetStorageManager
            storage_manager = DatasetStorageManager()
            
            # Load existing metadata to preserve fields like 'original_filename'
            metadata = storage_manager.load_metadata(self.user.id, self.dataset.id) or {}
            
            # Update specific fields
            metadata.update({
                'rows_count': len(df),
                'columns_count': len(df.columns),
                'file_size': os.path.getsize(full_path),
                'last_modified': pd.Timestamp.now().isoformat()
            })
            
            # Save metadata which automagically updates the preview_data when df is passed
            storage_manager.save_metadata(metadata, self.user.id, self.dataset.id, df=df)
            
        except Exception as e:
            logger.error(f"Failed to update dataset metadata/preview: {e}")
        
        logger.info(f"Tool '{self.name}' overwrote dataset {self.dataset.id} with new data (rows={len(df)}, cols={len(df.columns)})")
        
        return {"id": str(self.dataset.id), "name": self.dataset.name, "is_new": False}

    def load_dataset_stream(self, columns=None, batch_size: int = 100_000):
        """
        Stream a dataset in batches as pandas.DataFrame objects.

        This is helpful for very large datasets that cannot fit into memory.

        Yields:
            pandas.DataFrame: successive chunks of the dataset
        """
        from ...dataset_utils import load_dataset_stream

        self.validate_dataset_requirement()
        return load_dataset_stream(self.dataset.id, columns=columns, batch_size=batch_size)

    def strip_plotly_cdn(self, html: str) -> str:
        """
        Remove any <script> tags that attempt to load Plotly from external CDNs.
        This ensures the application uses the single, pinned Plotly script
        included in the main page template instead of per-artifact CDN loads.

        Args:
            html: HTML fragment potentially containing <script src="...plotly..."> tags

        Returns:
            Cleaned HTML string with CDN plotly script tags removed.
        """
        try:
            import re
            # Remove script tags that reference cdn.plot.ly or plotly-*.min.js
            html = re.sub(r'<script[^>]*src=["\'][^"\']*cdn\.plot\.ly[^"\']*["\'][^>]*>\s*</script>', '', html, flags=re.I)
            html = re.sub(r'<script[^>]*src=["\'][^"\']*plotly-[^"\']*["\'][^>]*>\s*</script>', '', html, flags=re.I)
            # Remove inline script blocks that contain explicit cdn.plot.ly references
            html = re.sub(r'<script[^>]*>[^<]*(?:cdn\.plot\.ly|plotly-[0-9\.]+\.min\.js)[^<]*</script>', '', html, flags=re.I|re.S)
            return html
        except Exception:
            # If cleaning fails for any reason, return original HTML to avoid breaking output
            logger.exception("strip_plotly_cdn failed")
            return html

    def clean_plotly_layout(self, fig) -> None:
        """
        Clean Plotly Figure layout to remove or unset axis 'matches' properties
        that can create circular references (and generate the repeated
        "ignored yaxis2.matches ... to avoid an infinite loop" warnings).

        This mutates the passed `fig` in-place.
        """
        try:
            # Use the figure JSON to inspect layout axis configs
            layout = fig.to_plotly_json().get('layout', {})
            for axis_key, axis_val in layout.items():
                if isinstance(axis_val, dict) and 'matches' in axis_val:
                    matches_value = axis_val.get('matches')
                    # Normalize axis_key to short form: 'yaxis2' -> 'y2', 'xaxis' -> 'x'
                    short = axis_key.replace('axis', '')
                    # If matches points back to the same axis (circular) or references a suspect id,
                    # unset it to avoid infinite-loop protection warnings from plotly.js.
                    if matches_value is None:
                        continue
                    if isinstance(matches_value, str) and (
                        matches_value == short or matches_value == axis_key or matches_value.endswith(short)
                    ):
                        try:
                            # Set matches to None for this axis
                            fig.update_layout(**{axis_key: {'matches': None}})
                        except Exception:
                            # defensive: try direct attribute set
                            try:
                                if hasattr(fig.layout, axis_key):
                                    getattr(fig.layout, axis_key).matches = None
                            except Exception:
                                logger.debug(f"Failed to unset matches for {axis_key}")
        except Exception:
            logger.exception("clean_plotly_layout failed")