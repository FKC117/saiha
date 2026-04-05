import logging
from django.http import HttpResponse
from django.urls import reverse

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet
from quantalytics.models import AnalysisResult

logger = logging.getLogger(__name__)

class WordReportExportTool(BaseAnalysisTool):
    """
    A tool to generate a .docx report from user-selected analysis sections.
    """
    name = "Export Word Report"
    tool_type = "word_report_export"
    description = "Generates a comprehensive .docx report with all analysis results from the current session."

    def get_parameters_schema(self) -> ToolParameterSet:
        """This tool requires no parameters."""
        return ToolParameterSet(tool_name=self.name)

    def execute(self, query: str = "", **kwargs) -> dict:
        """
        Triggers a full report download by returning a redirect instruction.
        """
        try:
            self.validate_dataset_requirement()
            results = AnalysisResult.objects.filter(session=self.session).order_by('created_at')
            if not results.exists():
                return {'status': 'error', 'summary': 'No analysis results found in this session to export.'}

            # Instead of generating the file here, we generate the URL to the download endpoint.
            download_url = reverse('quantalytics_api:download_full_report', args=[self.session.id])

            return {
                'status': 'redirect',
                'redirect_url': download_url,
                'summary': 'Report has been generated and exported to a Word file. Please check your download folder.'
            }
        except Exception as e:
            self.log_error(e)
            return {'status': 'error', 'summary': f"Error generating Word report: {str(e)}"}