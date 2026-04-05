import pandas as pd
import chardet
import json
import os
import re
import logging
from django.core.exceptions import ValidationError

logger = logging.getLogger(__name__)

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

    def sanitize_columns(self, df):
        """Sanitizes column names to be valid Python identifiers."""
        new_columns = {}
        seen_columns = {}
        for col in df.columns:
            sanitized_col = re.sub(r'[^0-9a-zA-Z_]', '_', str(col))
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

    def clean_dataframe(self, df):
        """Cleans the dataframe by stripping whitespace and handling empty rows/cols."""
        df = df.copy()
        df = self.sanitize_columns(df)
        
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace({'nan': None, 'None': None, '': None})
        
        df = df.dropna(how='all')
        df = df.reset_index(drop=True)
        return df

    def process_file(self, file):
        """Main method to process uploaded file and return a cleaned DataFrame."""
        self.validate_file(file)
        ext = os.path.splitext(file.name)[1].lower()
        
        try:
            if ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file, engine='openpyxl')
            elif ext == '.csv':
                # Try UTF-8 first, fallback to latin-1
                try:
                    df = pd.read_csv(file, encoding='utf-8')
                except UnicodeDecodeError:
                    file.seek(0)
                    df = pd.read_csv(file, encoding='latin-1')
            elif ext == '.json':
                df = pd.read_json(file)
            else:
                raise ValidationError("Unsupported file extension")
            
            df = self.clean_dataframe(df)
            
            self.metadata = {
                'file_type': ext.strip('.'),
                'rows_count': len(df),
                'columns_count': len(df.columns)
            }
            
            return df, self.metadata
            
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            raise ValidationError(f"Could not process file: {str(e)}")

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
