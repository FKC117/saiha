"""
Export Tool
Provides export options and recommendations for analysis results.
"""

import pandas as pd
import os
from typing import Dict, Any, List
from django.core.files.storage import default_storage
from .base_tool import BaseAnalysisTool


class ExportTool(BaseAnalysisTool):
    """Tool for exporting analysis results."""
    
    @property
    def name(self) -> str:
        return "export"
    
    @property
    def description(self) -> str:
        return "Export analysis results in various formats (CSV, JSON, PDF, Excel)"
    
    def execute(self, query: str, **kwargs) -> str:
        """Execute export recommendations."""
        try:
            self.validate_dataset_requirement()
            
            # Load dataset from storage
            with default_storage.open(self.dataset.processed_file_path, 'rb') as f:
                df = pd.read_parquet(f)
            
            # Analyze dataset for export recommendations
            export_context = self._analyze_export_context(df, query)
            
            # Generate export recommendations
            response = f"📁 **Export Options**\n\n"
            
            # Basic export information
            response += f"**Dataset Information:**\n"
            response += f"- Rows: {len(df):,}\n"
            response += f"- Columns: {len(df.columns)}\n"
            response += f"- File size: {self._estimate_file_size(df)}\n\n"
            
            # Available formats
            response += f"**Available Formats:**\n"
            response += self._get_format_recommendations(export_context)
            
            # Specific recommendations based on query
            response += self._get_query_specific_export_recommendations(query, export_context)
            
            # Export best practices
            response += self._get_export_best_practices(export_context)
            
            self.log_execution(query, response, success=True)
            return response
            
        except Exception as e:
            self.log_error(e)
            error_msg = f"Error providing export options: {str(e)}"
            self.log_execution(query, error_msg, success=False)
            return error_msg
    
    def _analyze_export_context(self, df: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Analyze dataset and query for export recommendations."""
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Analyze data complexity
        has_missing_data = df.isnull().any().any()
        has_duplicates = df.duplicated().any()
        has_large_text = any(df[col].astype(str).str.len().max() > 100 for col in categorical_columns)
        
        return {
            'rows': len(df),
            'columns': len(df.columns),
            'numeric_columns': len(numeric_columns),
            'categorical_columns': len(categorical_columns),
            'date_columns': len(date_columns),
            'has_missing_data': has_missing_data,
            'has_duplicates': has_duplicates,
            'has_large_text': has_large_text,
            'is_large_dataset': len(df) > 10000,
            'is_wide_dataset': len(df.columns) > 20,
            'query_intent': self._analyze_query_intent(query)
        }
    
    def _analyze_query_intent(self, query: str) -> str:
        """Analyze query to determine export intent."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['share', 'send', 'email', 'collaborate']):
            return 'sharing'
        elif any(word in query_lower for word in ['report', 'presentation', 'present']):
            return 'reporting'
        elif any(word in query_lower for word in ['backup', 'save', 'archive']):
            return 'backup'
        elif any(word in query_lower for word in ['api', 'integration', 'system']):
            return 'integration'
        else:
            return 'general'
    
    def _estimate_file_size(self, df: pd.DataFrame) -> str:
        """Estimate file size for different formats."""
        memory_usage = df.memory_usage(deep=True).sum()
        
        # Rough estimates for different formats
        csv_size = memory_usage * 1.2  # CSV is typically 20% larger
        excel_size = memory_usage * 1.5  # Excel with formatting
        json_size = memory_usage * 1.8  # JSON with metadata
        
        return f"~{memory_usage / 1024 / 1024:.1f} MB (CSV: {csv_size / 1024 / 1024:.1f} MB, Excel: {excel_size / 1024 / 1024:.1f} MB)"
    
    def _get_format_recommendations(self, context: Dict[str, Any]) -> str:
        """Get format recommendations based on context."""
        recommendations = ""
        
        # CSV recommendations
        recommendations += f"**CSV Format:**\n"
        recommendations += f"- Best for: Data analysis, machine learning, simple sharing\n"
        recommendations += f"- Pros: Universal compatibility, small file size, fast processing\n"
        recommendations += f"- Cons: No formatting, limited data types\n"
        
        # Excel recommendations
        if not context['is_large_dataset']:
            recommendations += f"\n**Excel Format:**\n"
            recommendations += f"- Best for: Business reports, presentations, manual review\n"
            recommendations += f"- Pros: Formatting, multiple sheets, charts, formulas\n"
            recommendations += f"- Cons: Larger file size, row limit (1M)\n"
        
        # JSON recommendations
        if context['has_large_text'] or context['query_intent'] == 'integration':
            recommendations += f"\n**JSON Format:**\n"
            recommendations += f"- Best for: APIs, web applications, complex data structures\n"
            recommendations += f"- Pros: Preserves data types, hierarchical data, metadata\n"
            recommendations += f"- Cons: Larger file size, less human-readable\n"
        
        # PDF recommendations
        if context['query_intent'] == 'reporting':
            recommendations += f"\n**PDF Format:**\n"
            recommendations += f"- Best for: Final reports, presentations, archival\n"
            recommendations += f"- Pros: Fixed formatting, professional appearance, universal viewing\n"
            recommendations += f"- Cons: Not editable, larger file size\n"
        
        return recommendations
    
    def _get_query_specific_export_recommendations(self, query: str, context: Dict[str, Any]) -> str:
        """Get specific export recommendations based on query."""
        recommendations = "\n**Specific Recommendations:**\n"
        
        if context['query_intent'] == 'sharing':
            if context['is_large_dataset']:
                recommendations += f"- Use CSV for large datasets (current: {context['rows']:,} rows)\n"
            else:
                recommendations += f"- Use Excel for easy sharing and collaboration\n"
            
            if context['has_missing_data']:
                recommendations += f"- Include data quality report with missing data summary\n"
        
        elif context['query_intent'] == 'reporting':
            recommendations += f"- Use PDF for final reports with charts and formatting\n"
            recommendations += f"- Include Excel backup for data verification\n"
            recommendations += f"- Add executive summary with key findings\n"
        
        elif context['query_intent'] == 'backup':
            recommendations += f"- Use CSV for data backup (smallest file size)\n"
            recommendations += f"- Include metadata file with column descriptions\n"
            recommendations += f"- Compress files for storage efficiency\n"
        
        elif context['query_intent'] == 'integration':
            recommendations += f"- Use JSON for API integration\n"
            recommendations += f"- Include data schema documentation\n"
            recommendations += f"- Validate data format before export\n"
        
        else:
            recommendations += f"- Start with CSV for data analysis\n"
            if not context['is_large_dataset']:
                recommendations += f"- Use Excel for formatted reports\n"
            recommendations += f"- Consider JSON for complex data structures\n"
        
        return recommendations
    
    def _get_export_best_practices(self, context: Dict[str, Any]) -> str:
        """Get export best practices."""
        practices = "\n**Export Best Practices:**\n"
        
        # Data preparation
        if context['has_missing_data']:
            practices += f"- Clean missing data before export\n"
        
        if context['has_duplicates']:
            practices += f"- Remove or flag duplicate rows\n"
        
        # File organization
        practices += f"- Use descriptive filenames with timestamps\n"
        practices += f"- Include data dictionary for complex datasets\n"
        
        # Security considerations
        if context['rows'] > 1000:
            practices += f"- Consider data privacy and security requirements\n"
            practices += f"- Use secure transfer methods for sensitive data\n"
        
        # Performance considerations
        if context['is_large_dataset']:
            practices += f"- Consider splitting large datasets into chunks\n"
            practices += f"- Use compression for file transfer\n"
        
        return practices
