import os
import django
import sys
import time
from uuid import uuid4

# Setup Django
sys.path.append('f:/saiha/alaina')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'alaina.settings')
django.setup()

from saiha.agents.param_corrector import ParamCorrector
from saiha.agents.analysis_planner import analysis_planner
from saiha.models import AnalysisSession, Dataset, User
from saiha.analysis_tools.registry import tool_registry

def test_param_corrector_synonyms():
    print("\n--- Testing ParamCorrector Synonyms ---")
    columns = ["Age", "Annual_Income", "Gender"]
    corrector = ParamCorrector(columns)
    
    # Mock Schema for Histogram (expects 'variable')
    tool = tool_registry.get_tool("histogram")
    schema = tool.get_parameters_schema()
    
    # LLM hallucinates 'column_name' instead of 'variable'
    raw_params = {"column_name": "income"}
    
    corrected = corrector.apply_to_params(raw_params, schema)
    print(f"Original: {raw_params}")
    print(f"Corrected: {corrected}")
    
    assert "variable" in corrected
    assert corrected["variable"] == "Annual_Income"
    print("SUCCESS: Synonym Mapping Success!")

def test_planner_performance():
    print("\n--- Testing Planner Performance (Large Schema) ---")
    user = User.objects.first()
    dataset = Dataset.objects.first()
    
    # Create a mock session with 500 column names in memory (not real DB columns for speed)
    session = AnalysisSession.objects.create(
        user=user,
        dataset=dataset,
        working_memory={"active_columns": ["Existing_Col"]}
    )
    
    # Mock columns
    mock_columns = [f"Col_{i}" for i in range(500)]
    
    # Monkey-patch _get_dataset_columns for this test instance
    original_get = analysis_planner._get_dataset_columns
    analysis_planner._get_dataset_columns = lambda s: mock_columns
    
    start_time = time.time()
    # Test query that has NO match (worst case for nested fuzzy matching)
    has_signal = analysis_planner._query_has_new_signal("summarize something totally unrelated", session)
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"Time taken for 500 columns: {duration:.4f}s")
    
    # Restore
    analysis_planner._get_dataset_columns = original_get
    session.delete()
    
    assert duration < 0.1, f"Performance test failed: {duration:.4f}s is too slow!"
    print("SUCCESS: Performance Optimization Success!")

if __name__ == "__main__":
    try:
        test_param_corrector_synonyms()
        test_planner_performance()
        print("\n[SUCCESS] ALL TESTS PASSED")
    except Exception as e:
        print(f"\n[ERROR] TEST FAILED: {e}")
        sys.exit(1)
