import logging
from typing import Dict, Any
import pandas as pd
import numpy as np

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType

logger = logging.getLogger(__name__)

class DataTypeConversionTool(BaseAnalysisTool):
    """
    Tool for converting column data types (Numeric, Categorical, Boolean).
    Note: Datetime conversions are handled by DatetimeConversionTool.
    """
    
    name = "data_type_conversion"
    description = "Convert column execution types (e.g., String to Numeric, Numeric to Categorical)."
    
    def get_parameters_schema(self) -> ToolParameterSet:
        params = ToolParameterSet(tool_name=self.name)
        
        # 1. Column Selection
        params.add_parameter(ToolParameter(
            name="column_to_convert",
            parameter_type=ParameterType.COLUMN_SELECT,
            label="Column to Convert",
            description="Select the column you want to convert.",
            required=True,
            column_source="all"
        ))
        
        # 2. Target Type Selection
        params.add_parameter(ToolParameter(
            name="target_type",
            parameter_type=ParameterType.SELECT,
            label="Target Data Type",
            description="Select the new data type for the column.",
            required=True,
            options=[
                {"value": "numeric", "label": "Numeric (Integer/Float)"},
                {"value": "categorical", "label": "Text / Categorical"},
                {"value": "boolean", "label": "Boolean (True/False)"}
            ],
            default_value="numeric"
        ))
        
        # 3. Error Handling
        params.add_parameter(ToolParameter(
            name="handle_errors",
            parameter_type=ParameterType.SELECT,
            label="Error Handling",
            description="How to handle values that cannot be converted.",
            required=True,
            options=[
                {"value": "coerce", "label": "Coerce (Set errors to NaN)"},
                {"value": "raise", "label": "Raise Error (Stop execution)"},
                {"value": "ignore", "label": "Ignore (Keep original values)"}
            ],
            default_value="coerce"
        ))
        
        return params

    def execute(self, query: str = "", **kwargs) -> Dict[str, Any]:
        """
        Execute the data type conversion.
        """
        try:
            # Extract parameters
            column = kwargs.get('column_to_convert')
            target_type = kwargs.get('target_type')
            handle_errors = kwargs.get('handle_errors', 'coerce')
            
            if not column or not target_type:
                return {"error": "Missing required parameters: column_to_convert or target_type"}
                
            df = self.load_dataset()
            if df is None:
                return {"error": "Dataset not found"}
            
            if column not in df.columns:
                 return {"error": f"Column '{column}' not found in dataset."}
                
            original_dtype = str(df[column].dtype)
            report = {}
            
            try:
                if target_type == 'numeric':
                    # Clean currency/percentage symbols if common string
                    if df[column].dtype == 'object':
                        df[column] = df[column].astype(str).str.replace(r'[$,%]', '', regex=True)
                    
                    df[column] = pd.to_numeric(df[column], errors=handle_errors)
                    
                elif target_type in ['categorical', 'text', 'string']:
                    df[column] = df[column].astype(str)
                    
                elif target_type == 'boolean':
                    # Simple mapping for common Yes/No
                    bool_map = {'yes': True, 'no': False, 'true': True, 'false': False, '1': True, '0': False}
                    if df[column].dtype == 'object':
                            df[column] = df[column].str.lower().map(bool_map).fillna(False)
                    else:
                            df[column] = df[column].astype(bool)
                            
                else:
                    return {"error": f"Unknown target type: {target_type}"}

                report = {
                    "column": column, 
                    "status": "success", 
                    "from": original_dtype, 
                    "to": str(df[column].dtype) 
                }
                
            except Exception as e:
                    return {"error": f"Conversion failed: {str(e)}"}
            
            # Save changes
            self.save_dataset(df)
            
            return {
                "status": "success",
                "report": [report],
                "message": f"Successfully converted '{column}' to {target_type}."
            }
            
        except Exception as e:
            logger.error(f"Data type conversion failed: {e}")
            return {"error": str(e)}

    def interpret(self, result: Dict[str, Any]) -> str:
        if 'error' in result:
             return f"<p class='text-danger'>Error: {result['error']}</p>"
             
        report_html = "<ul>"
        for item in result.get('report', []):
            msg = f"{item['column']}: {item['status']} ({item['from']} → {item['to']})"
            report_html += f"<li style='color:green'>{msg}</li>"
        report_html += "</ul>"
        
        return f"<h3>Conversion Success</h3>{report_html}"
