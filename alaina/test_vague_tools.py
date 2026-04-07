import os
import django
import uuid

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'alaina.settings')
django.setup()

from saiha.models import AnalysisSession
from saiha.agents.analysis_agent import get_analysis_agent
from saiha.analysis_tools.box_plot_tool import BoxPlotTool
from saiha.analysis_tools.histogram_tool import HistogramTool
from saiha.analysis_tools.outlier_detection_tool import OutlierDetectionTool
from saiha.analysis_tools.scatter_plot_tool import ScatterPlotTool

def test_vague_execution():
    session = AnalysisSession.objects.filter(dataset__isnull=False).first()
    if not session:
        print("No session with dataset found. Cannot test.")
        return

    print(f"Testing with Session: {session.id} (Dataset: {session.dataset.name})")
    
    # Initialize the real agent to provide context to tools
    agent = get_analysis_agent(str(session.id))

    tools = [
        (BoxPlotTool, "Box Plot"),
        (HistogramTool, "Histogram"),
        (OutlierDetectionTool, "Outlier Detection"),
        (ScatterPlotTool, "Scatter Plot")
    ]

    for tool_class, label in tools:
        print(f"\n--- Testing {label} ---")
        tool = tool_class(agent=agent)
        result = tool.execute(query="show me charts")
        
        status = result.get('status')
        artifacts_count = len(result.get('artifacts', []))
        summary = result.get('summary')
        
        print(f"Status: {status}")
        print(f"Artifacts: {artifacts_count}")
        print(f"Summary: {summary}")
        
        if status != 'ok' or artifacts_count == 0:
            print(f"FAILED: {label}")
        else:
            print(f"PASSED: {label}")

if __name__ == "__main__":
    test_vague_execution()
