import os
import sys
import numpy as np
import pandas as pd
from typing import Any, Dict

# Mock the environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'alaina.settings')
import django
django.setup()

from saiha.analysis_tools.base_tool import BaseAnalysisTool

class TestTool(BaseAnalysisTool):
    def name(self) -> str: return "test_tool"
    def execute(self, query: str = "", **kwargs: Any) -> dict:
        return {
            "status": "ok",
            "summary": "Test with NaN",
            "data": {
                "val": np.nan,
                "inf": np.inf,
                "list": [1.0, np.nan, 2.0],
                "dict": {"a": np.nan, "b": 3.0}
            },
            "artifacts": [
                {"type": "table", "title": "NaN Table", "data": [[np.nan, 1], [2, np.inf]]}
            ]
        }

def verify_json_safety():
    tool = TestTool()
    # execute
    raw_result = tool.execute()
    # validate_and_run calls _normalize_legacy_result which calls sanitize_json_data
    normalized = tool._normalize_legacy_result(raw_result)
    
    print("Normalized Result Data:", normalized.data)
    print("Normalized Artifacts:", normalized.artifacts)
    
    # Test JSON serialization (which usually fails for NaN)
    import json
    try:
        json_str = json.dumps(normalized.dict())
        print("JSON Serialization SUCCESS")
        if "null" in json_str:
            print("Verified: NaN/Inf converted to null")
    except Exception as e:
        print(f"JSON Serialization FAILED: {e}")

if __name__ == "__main__":
    verify_json_safety()
