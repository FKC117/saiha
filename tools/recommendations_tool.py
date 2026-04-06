"""
Recommendations Tool
Provides intelligent analysis recommendations based on dataset characteristics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from django.core.files.storage import default_storage
from .base_tool import BaseAnalysisTool


class RecommendationsTool(BaseAnalysisTool):
    """Tool for providing analysis recommendations."""
    
    @property
    def name(self) -> str:
        return "recommendations"
    
    @property
    def description(self) -> str:
        return "Get intelligent recommendations for what analysis to perform next based on the dataset characteristics"
    
    def execute(self, query: str, **kwargs) -> str:
        """Execute recommendations analysis."""
        try:
            self.validate_dataset_requirement()
            
            # Load dataset from storage
            with default_storage.open(self.dataset.processed_file_path, 'rb') as f:
                df = pd.read_parquet(f)
            
            # Analyze dataset characteristics
            analysis_context = self._analyze_dataset_characteristics(df)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(analysis_context, query)
            
            # Format response
            response = f"💡 **Analysis Recommendations**\n\n"
            response += f"**Dataset Summary:**\n"
            response += f"- Total columns: {analysis_context['total_columns']}\n"
            response += f"- Numeric columns: {analysis_context['numeric_columns']}\n"
            response += f"- Categorical columns: {analysis_context['categorical_columns']}\n"
            response += f"- Date columns: {analysis_context['date_columns']}\n"
            response += f"- Rows: {analysis_context['rows']:,}\n\n"
            
            response += f"**Recommended Analyses:**\n"
            for i, rec in enumerate(recommendations, 1):
                response += f"{i}. **{rec['name']}** ({rec['type']})\n"
                response += f"   {rec['description']}\n"
                if 'columns' in rec:
                    response += f"   Columns: {', '.join(rec['columns'][:3])}\n"
                response += "\n"
            
            # Add priority recommendations
            response += self._get_priority_recommendations(analysis_context, query)
            
            self.log_execution(query, response, success=True)
            return response
            
        except Exception as e:
            self.log_error(e)
            error_msg = f"Error getting recommendations: {str(e)}"
            self.log_execution(query, error_msg, success=False)
            return error_msg
    
    def _analyze_dataset_characteristics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze dataset characteristics for recommendations."""
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Analyze data quality
        missing_percentage = (df.isnull().sum().sum() / df.size) * 100
        duplicate_percentage = (df.duplicated().sum() / len(df)) * 100
        
        # Analyze column distributions
        numeric_distributions = {}
        for col in numeric_columns:
            numeric_distributions[col] = {
                'skewness': df[col].skew(),
                'kurtosis': df[col].kurtosis(),
                'outliers': self._count_outliers(df[col])
            }
        
        return {
            'total_columns': len(df.columns),
            'numeric_columns': len(numeric_columns),
            'categorical_columns': len(categorical_columns),
            'date_columns': len(date_columns),
            'rows': len(df),
            'missing_percentage': missing_percentage,
            'duplicate_percentage': duplicate_percentage,
            'numeric_distributions': numeric_distributions,
            'has_time_series': len(date_columns) > 0,
            'has_correlations': len(numeric_columns) > 1
        }
    
    def _count_outliers(self, series: pd.Series) -> int:
        """Count outliers in a numeric series."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return len(series[(series < lower_bound) | (series > upper_bound)])
    
    def _generate_recommendations(self, context: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
        """Generate analysis recommendations based on context."""
        recommendations = []
        
        # Statistical analysis recommendations
        if context['numeric_columns'] >= 1:
            recommendations.append({
                'type': 'statistical',
                'name': 'Descriptive Statistics',
                'description': f'Analyze {context["numeric_columns"]} numeric columns with basic statistics',
                'priority': 'high'
            })
        
        # Correlation analysis recommendations
        if context['numeric_columns'] >= 2:
            recommendations.append({
                'type': 'correlation',
                'name': 'Correlation Analysis',
                'description': f'Find relationships between {context["numeric_columns"]} numeric variables',
                'priority': 'high'
            })
        
        # Categorical analysis recommendations
        if context['categorical_columns'] >= 1:
            recommendations.append({
                'type': 'categorical',
                'name': 'Categorical Analysis',
                'description': f'Analyze {context["categorical_columns"]} categorical columns',
                'priority': 'medium'
            })
        
        # Time series analysis recommendations
        if context['has_time_series'] and context['numeric_columns'] >= 1:
            recommendations.append({
                'type': 'time_series',
                'name': 'Time Series Analysis',
                'description': f'Analyze trends over time with {context["date_columns"]} date columns',
                'priority': 'high'
            })
        
        # Data quality recommendations
        if context['missing_percentage'] > 5 or context['duplicate_percentage'] > 1:
            recommendations.append({
                'type': 'data_quality',
                'name': 'Data Quality Assessment',
                'description': f'Address data quality issues (missing: {context["missing_percentage"]:.1f}%, duplicates: {context["duplicate_percentage"]:.1f}%)',
                'priority': 'high'
            })
        
        # Visualization recommendations
        if context['numeric_columns'] >= 1 or context['categorical_columns'] >= 1:
            recommendations.append({
                'type': 'visualization',
                'name': 'Data Visualization',
                'description': f'Create visualizations for {context["numeric_columns"]} numeric and {context["categorical_columns"]} categorical columns',
                'priority': 'medium'
            })
        
        # Machine learning recommendations
        if context['numeric_columns'] >= 3 and context['rows'] > 100:
            recommendations.append({
                'type': 'machine_learning',
                'name': 'Machine Learning Analysis',
                'description': f'Apply ML algorithms to {context["rows"]:,} rows with {context["numeric_columns"]} features',
                'priority': 'low'
            })
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def _get_priority_recommendations(self, context: Dict[str, Any], query: str) -> str:
        """Get priority recommendations based on context and query."""
        priority_recs = "\n**Priority Recommendations:**\n"
        
        # High priority based on data quality
        if context['missing_percentage'] > 10:
            priority_recs += "🔴 **URGENT**: High missing data percentage - clean data first\n"
        
        if context['duplicate_percentage'] > 5:
            priority_recs += "🔴 **URGENT**: High duplicate percentage - remove duplicates\n"
        
        # Medium priority based on data characteristics
        if context['numeric_columns'] >= 2 and not context['has_correlations']:
            priority_recs += "🟡 **RECOMMENDED**: Explore relationships between numeric variables\n"
        
        if context['has_time_series'] and context['numeric_columns'] >= 1:
            priority_recs += "🟡 **RECOMMENDED**: Time series analysis for trend identification\n"
        
        # Low priority suggestions
        if context['rows'] > 1000 and context['numeric_columns'] >= 3:
            priority_recs += "🟢 **OPTIONAL**: Consider machine learning for pattern discovery\n"
        
        return priority_recs
