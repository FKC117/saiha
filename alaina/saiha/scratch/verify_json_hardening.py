
import sys
import os
import django
import pandas as pd
import numpy as np
import json

# Setup Django Environment
sys.path.append('f:/saiha/alaina')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'alaina.settings')
django.setup()

from saiha.analysis_tools.base_tool import BaseAnalysisTool

class TestTool(BaseAnalysisTool):
    def execute(self, query="", **kwargs):
        return {"status": "ok", "data": {"val": np.nan}}

def test_sanitization():
    tool = TestTool()
    
    dirty_data = {
        "numeric_nan": np.nan,
        "numeric_inf": np.inf,
        "nested_list": [1.0, np.nan, 2.0],
        "nested_dict": {"a": np.nan, "b": "hello"},
        "numpy_array": np.array([1, 2, np.nan]),
        "pandas_series": pd.Series([1, 2, np.nan]),
        "mixed": [{"val": np.nan}, np.array([np.inf])]
    }
    
    print("--- Original Data (with NaNs) ---")
    print(dirty_data)
    
    clean_data = tool.sanitize_json_data(dirty_data)
    
    print("\n--- Cleaned Data (JSON Safe) ---")
    print(clean_data)
    
    try:
        json_str = json.dumps(clean_data)
        print("\n--- JSON Serialization Test ---")
        print("SUCCESS: JSON is safe.")
        print(json_str)
    except Exception as e:
        print(f"\nFAILURE: JSON serialization failed: {e}")

if __name__ == "__main__":
    test_sanitization()
