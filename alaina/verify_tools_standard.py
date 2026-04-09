import os
import sys
import pandas as pd
import numpy as np
from typing import Any, Dict

# Mock Django settings for testing
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'alaina.settings')
import django
django.setup()

from saiha.analysis_tools.column_analysis_tool import ColumnAnalysisTool
from saiha.analysis_tools.statistical_analysis_tool import StatisticalAnalysisTool
from saiha.analysis_tools.box_plot_tool import BoxPlotTool

# Create dummy data which causes issues (constant values)
df = pd.DataFrame({
    'col_a': [1, 1, 1, 1, 1], # Constant
    'col_b': [1, 2, 3, 4, 5], # Normal
    'col_c': ['A', 'A', 'B', 'B', 'C'] # Categorical
})

class MockAgent:
    def __init__(self):
        self.session = None
        self.dataset = None
        self.user = None

def test_tool(tool_class, params):
    print(f"\n--- Testing {tool_class.__name__} ---")
    tool = tool_class(agent=MockAgent())
    # Mock load_dataset
    tool.load_dataset = lambda: df
    
    try:
        results = tool.execute(**params)
        print(f"Status: {results.get('status')}")
        print(f"Artifacts Count: {len(results.get('artifacts', []))}")
        
        # Test interpret
        interpretation = tool.interpret(results)
        print(f"Interpretation:\n{interpretation}")
        
        return results
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Test ColumnAnalysisTool
    test_tool(ColumnAnalysisTool, {"columns_to_analyze": ["col_a", "col_b", "col_c"], "generate_plots": True})
    
    # Test StatisticalAnalysisTool
    test_tool(StatisticalAnalysisTool, {"columns": ["col_a", "col_b"]})
    
    # Test BoxPlotTool
    test_tool(BoxPlotTool, {"numeric_variable": "col_a"}) # Should skip plotting col_a
