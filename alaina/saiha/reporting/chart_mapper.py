import logging
import io
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, Tuple
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE

logger = logging.getLogger(__name__)

class ChartMapper:
    """
    The Hybrid Visualization Engine.
    Maps ECharts JSON → (Native PPT Chart | Static Image).
    Native Charting allows user editing in PowerPoint.
    Image Fallback ensures fidelity for complex data.
    """
    
    @staticmethod
    def map_to_pptx_chart(artifact: Dict[str, Any]) -> Optional[Tuple[Any, Any]]:
        """
        Attempts to map artifact to a native PPT chart type.
        Returns: (XL_CHART_TYPE, CategoryChartData) or None.
        """
        option = artifact.get('option', {})
        chart_type = artifact.get('id', '').split('_')[0] or 'bar'
        
        # We only support Bar, Line, Pie for Native Mapping to avoid UI/PPT breakage.
        if chart_type not in ['bar', 'line', 'pie', 'count', 'dist']:
            return None
        
        try:
            chart_data = CategoryChartData()
            
            # Extract Labels (Categories)
            labels = option.get('xAxis', {}).get('data', [])
            if not labels and chart_type == 'pie':
                series_data = option.get('series', [{}])[0].get('data', [])
                labels = [d.get('name') for d in series_data]
            
            if not labels: return None
            chart_data.categories = labels
            
            # Extract Series (Values)
            series_list = option.get('series', [])
            for s in series_list:
                name = s.get('name', 'Value')
                data = s.get('data', [])
                if chart_type == 'pie' and isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                    # Pie chart data is often [{name: x, value: y}]
                    data = [d.get('value', 0) for d in data]
                
                chart_data.add_series(name, data)
            
            # Map XL Types
            if chart_type in ['bar', 'count', 'dist']:
                return XL_CHART_TYPE.COLUMN_CLUSTERED, chart_data
            if chart_type == 'line':
                return XL_CHART_TYPE.LINE, chart_data
            if chart_type == 'pie':
                return XL_CHART_TYPE.PIE, chart_data
                
        except Exception as e:
            logger.error(f"Native Chart Mapping failed for {chart_type}: {e}")
            return None
        
        return None

    @staticmethod
    def generate_static_image(artifact: Dict[str, Any]) -> Optional[io.BytesIO]:
        """
        Server-side fallback for complex charts using matplotlib.
        """
        option = artifact.get('option', {})
        chart_id = artifact.get('id', '')
        
        try:
            plt.figure(figsize=(10, 6), dpi=100)
            plt.style.use('ggplot')
            
            if 'heatmap' in chart_id or 'correlation' in chart_id:
                # Heatmap Fallback
                data = option.get('series', [{}])[0].get('data', [])
                columns = option.get('xAxis', {}).get('data', [])
                matrix_size = len(columns)
                matrix = [[0] * matrix_size for _ in range(matrix_size)]
                for d in data:
                    matrix[d[1]][d[0]] = d[2]
                
                sns.heatmap(matrix, xticklabels=columns, yticklabels=columns, annot=True, cmap='RdBu_r', center=0)
                plt.title(artifact.get('title', 'Correlation Heatmap'))
            
            else:
                # Generic fallback (Title only)
                plt.text(0.5, 0.5, f"Consulting-Grade Visual: {artifact.get('title')}\n[Data in Appendix]", 
                         ha='center', va='center')
            
            img_stream = io.BytesIO()
            plt.savefig(img_stream, format='png', bbox_inches='tight')
            plt.close()
            img_stream.seek(0)
            return img_stream
            
        except Exception as e:
            logger.error(f"Static Image generation failed: {e}")
            return None
