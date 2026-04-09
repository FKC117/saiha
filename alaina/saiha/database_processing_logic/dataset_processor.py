import pandas as pd
import chardet
import json
import os
import re
import logging
from django.core.exceptions import ValidationError

logger = logging.getLogger(__name__)

import functools
import time

logger = logging.getLogger(__name__)

class EmptyColumnsDetected(ValidationError):
    """Raised when a dataset contains columns that are entirely NULL."""
    def __init__(self, message, columns=None):
        super().__init__(message)
        self.columns = columns or []

def instrument_step(func):
    """Decorator to measure and log the execution time of processing steps."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        duration = end_time - start_time
        logger.info(f"PERF: {func.__name__} completed in {duration:.4f}s")
        return result
    return wrapper

class DatasetProcessor:
    """
    Handles dataset file processing including encoding detection,
    data cleaning, and metadata extraction.
    Ported and simplified from Quantly.
    """
    
    ALLOWED_EXTENSIONS = ['.xlsx', '.xls', '.csv', '.json']
    DEFAULT_MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB default
    
    def __init__(self):
        self.metadata = {}
    
    @instrument_step
    def validate_file(self, file):
        """Validates uploaded file before processing."""
        if file.size == 0:
            raise ValidationError("File is empty")
        
        if file.size > self.DEFAULT_MAX_FILE_SIZE:
            raise ValidationError(f"File size exceeds limit")
        
        file_name = file.name.lower()
        if not any(file_name.endswith(ext) for ext in self.ALLOWED_EXTENSIONS):
            raise ValidationError(f"Unsupported file format. Allowed: {', '.join(self.ALLOWED_EXTENSIONS)}")
        
        return True

    @instrument_step
    def sanitize_columns(self, df):
        """Sanitizes column names to be valid Python identifiers."""
        new_columns = {}
        seen_columns = {}
        for col in df.columns:
            # Strip whitespace from column name before sanitizing
            col_str = str(col).strip()
            sanitized_col = re.sub(r'[^0-9a-zA-Z_]', '_', col_str)
            if sanitized_col and sanitized_col[0].isdigit():
                sanitized_col = '_' + sanitized_col
            sanitized_col = re.sub(r'__+', '_', sanitized_col).strip('_')
            
            original_sanitized = sanitized_col
            count = seen_columns.get(original_sanitized, 0)
            if count > 0:
                sanitized_col = f"{original_sanitized}_{count}"
            seen_columns[original_sanitized] = count + 1
            new_columns[col] = sanitized_col
        
        return df.rename(columns=new_columns)

    @instrument_step
    def clean_dataframe(self, df, drop_empty=False):
        """
        Cleans the dataframe by stripping whitespace and handling empty rows/cols.
        Hardened to detect entirely NULL columns and use modern nullable types.
        """
        df = df.copy()
        
        # 1. Strip column names first
        df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]
        
        # 2. Detect and handle entirely empty columns (Quantly Defense Layer)
        empty_cols = [col for col in df.columns if df[col].isnull().all()]
        if empty_cols:
             if drop_empty:
                 logger.info(f"DEFENSE: Dropping entirely empty columns as requested: {empty_cols}")
                 df = df.drop(columns=empty_cols)
             else:
                 raise EmptyColumnsDetected(f"Dataset contains entirely empty columns.", columns=empty_cols)

        # 3. Sanitize to valid Python identifiers
        df = self.sanitize_columns(df)
        
        # 4. Smart Whitespace Stripping & Null Handling
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
                except Exception:
                    pass
                
                # Replace common null representations
                df[col] = df[col].replace({'nan': None, 'None': None, '': None, 'NA': None, 'N/A': None})

        # 5. Numeric Coercion
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # errors='ignore' ensures we don't break non-numeric columns
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except Exception:
                    pass
        
        # 6. Modern Data Types (Use nullable string type for everything else that is 'object')
        # This provides better NA handling and performance
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = df[col].astype("string")
                except Exception:
                    pass

        # 7. Final Row Cleanup
        df = df.dropna(how='all')
        df = df.reset_index(drop=True)
        return df

    @instrument_step
    def process_file(self, file, drop_empty=False):
        """Main method to process uploaded file and return a cleaned DataFrame."""
        self.validate_file(file)
        ext = os.path.splitext(file.name)[1].lower()
        
        try:
            if ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file, engine='openpyxl')
            elif ext == '.csv':
                # Robust Encoding Detection (Quantly Defense Layer)
                df = self._read_csv_with_encoding(file)
            elif ext == '.json':
                df = pd.read_json(file)
            else:
                raise ValidationError("Unsupported file extension")
            
            df = self.clean_dataframe(df, drop_empty=drop_empty)
            
            self.metadata = {
                'file_type': ext.strip('.'),
                'rows_count': len(df),
                'columns_count': len(df.columns)
            }
            
            return df, self.metadata
            
        except EmptyColumnsDetected:
            raise # Re-raise specialized exception
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            raise ValidationError(f"Could not process file: {str(e)}")

    def _read_csv_with_encoding(self, file):
        """Detects encoding and reads CSV with multiple fallback attempts."""
        # 1. Detect encoding
        file.seek(0)
        raw_data = file.read(10000) # Sample for detection
        detection = chardet.detect(raw_data)
        detected_encoding = detection.get('encoding', 'utf-8')
        confidence = detection.get('confidence', 0)
        
        encodings_to_try = [detected_encoding, 'utf-8', 'cp1252', 'iso-8859-1', 'latin-1']
        
        for enc in encodings_to_try:
            if not enc: continue
            try:
                file.seek(0)
                logger.info(f"Attempting to read CSV with encoding: {enc} (Detection Confidence: {confidence:.2f})")
                return pd.read_csv(file, encoding=enc)
            except (UnicodeDecodeError, TypeError):
                continue
        
        raise ValidationError("Could not determine CSV encoding even with multiple fallbacks.")

    def get_column_metadata(self, df):
        """Extracts metadata for each column for database storage."""
        columns_metadata = []
        for i, col in enumerate(df.columns):
            col_data = df[col]
            
            # Simple type mapping
            dtype_str = str(col_data.dtype).lower()
            if 'int' in dtype_str:
                data_type = 'integer'
            elif 'float' in dtype_str:
                data_type = 'float'
            elif 'bool' in dtype_str:
                data_type = 'boolean'
            elif 'datetime' in dtype_str:
                data_type = 'date'
            else:
                data_type = 'string'
            
            columns_metadata.append({
                'column_name': col,
                'column_index': i,
                'data_type': data_type,
                'null_count': int(col_data.isnull().sum()),
                'unique_count': int(col_data.nunique()),
                'sample_values': col_data.dropna().head(5).tolist()
            })
        return columns_metadata
