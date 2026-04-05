import logging
from django.urls import reverse

from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameterSet
from saiha.models import AnalysisResult

logger = logging.getLogger(__name__)

class PowerPointReportExportTool(BaseAnalysisTool):
    """
    A tool to generate a .pptx report from all analysis sections in the current session.
    """
    name = "Export PowerPoint Report"
    tool_type = "powerpoint_report_export"
    description = "Generates a comprehensive .pptx presentation with all analysis results from the current session."

    def get_parameters_schema(self) -> ToolParameterSet:
        """This tool requires no parameters."""
        return ToolParameterSet(tool_name=self.name)

    def execute(self, query: str = "", **kwargs) -> dict:
        """Triggers a full report download by returning a redirect instruction."""
        try:
            self.validate_dataset_requirement()
            if not AnalysisResult.objects.filter(session=self.session).exists():
                return {'status': 'error', 'summary': 'No analysis results found in this session to export.'}

            download_url = reverse('quantalytics_api:download_full_pptx_report', args=[self.session.id])
            return {'status': 'redirect', 'redirect_url': download_url,
                    'summary': 'Presentation has been generated. Please check your download folder.'}
        except Exception as e:
            self.log_error(e)
            return {'status': 'error', 'summary': f"Error generating PowerPoint report: {str(e)}"}