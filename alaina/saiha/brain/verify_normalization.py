import os
import django
import sys

# Setup Django
sys.path.append(r"f:\saiha\alaina")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "alaina.settings")
django.setup()

from saiha.analysis_tools.base_tool import BaseAnalysisTool, ToolResult

class MockTool(BaseAnalysisTool):
    def execute(self, query="", **kwargs):
        return {
            "status": "ok",
            "message": "Success",
            "data": {
                "my_table": {
                    "records": [{"col1": 1, "col2": "a"}, {"col1": 2, "col2": "b"}]
                }
            }
        }

tool = MockTool()
result = tool.validate_and_run(None, {})
print(f"Status: {result.status}")
print(f"Artifacts Count: {len(result.artifacts)}")
if result.artifacts:
    print(f"First Artifact Type: {result.artifacts[0].get('type')}")
    print(f"First Artifact Title: {result.artifacts[0].get('title')}")

# Test Boxplot sorting priority
class SortTool(BaseAnalysisTool):
    def execute(self, query="", **kwargs):
        return {
            "status": "ok",
            "artifacts": [
                {"type": "plot", "title": "A Plot"},
                {"type": "table", "title": "A Table"}
            ]
        }
tool2 = SortTool()
result2 = tool2.validate_and_run(None, {})
print(f"First Artifact (Sorted): {result2.artifacts[0].get('type')}")
