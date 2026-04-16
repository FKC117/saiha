import os
import django
import sys
from unittest.mock import MagicMock

# Setup Django
sys.path.append(r"f:\saiha\alaina")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "alaina.settings")
django.setup()

from saiha.analysis_tools.base_tool import BaseAnalysisTool
from saiha.celery_tasks.analysis_tasks import execute_analysis_task
from saiha.models import AnalysisResult, AnalysisSession

print("--- Elite v3.8 Verification ---")

# 1. Mock Tool Result
mock_result_obj = MagicMock()
mock_result_obj.status = "success"
mock_result_obj.data = {"raw_table": {"records": [{"a": 1}]}}
mock_result_obj.artifacts = [{"type": "plot", "title": "Plot 1"}]
mock_result_obj.message = "Done"

# 2. Mock Tool Instance
mock_tool = MagicMock()
mock_tool.validate_and_run.return_value = mock_result_obj
mock_tool.sanitize_json_data = lambda x: x
mock_tool.load_dataset.return_value = None

# Injection logic for the task
import saiha.celery_tasks.analysis_tasks as tasks
from saiha.analysis_tools.registry import tool_registry

# Mock registry to return our mock tool
tool_registry.get_tool = MagicMock(return_value=mock_tool)

# 3. Create dummy session/result
session = AnalysisSession.objects.first()
if not session:
    print("No session found to test with.")
    sys.exit(0)

result_record = AnalysisResult.objects.create(
    session=session,
    tool_used="descriptive_statistics",
    query="test logic"
)

# 4. Trigger Task logic (manually call function for speed)
# We mock notification to avoid WebSocket errors in script
tasks.send_ws_notification = MagicMock()
tasks.interpretation_agent = MagicMock()

try:
    print(f"Executing task for result {result_record.id}...")
    execute_analysis_task.delay_on_commit = False # Run synchronously
    # Call directly to test the function logic
    execute_analysis_task(None, str(result_record.id), str(session.id), "descriptive_statistics", {}, "test-dedup", "test query")
    
    # 5. Verify Persistence
    result_record.refresh_from_db()
    data = result_record.result_data
    print(f"Status: {result_record.status}")
    print(f"Final Artifacts Key Count: {len(data.get('artifacts', []))}")
    
    # We expect 2 artifacts: the original plot AND the promoted table from 'raw_table'
    for i, art in enumerate(data.get('artifacts', [])):
        print(f"Artifact {i+1}: Type={art.get('type')}, Title={art.get('title')}")

except Exception as e:
    print(f"Test Failed: {e}")
    import traceback
    traceback.print_exc()
finally:
    result_record.delete()
