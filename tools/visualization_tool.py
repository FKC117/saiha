"""
Visualization Tool
Creates visualization recommendations and generates appropriate charts for data analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from django.core.files.storage import default_storage
from .base_tool import BaseAnalysisTool


class VisualizationTool(BaseAnalysisTool):
    """Tool for creating data visualizations."""
    
    @property
    def name(self) -> str:
        return "visualization"
    
    @property
    def description(self) -> str:
        return "Create visualizations including histograms, scatter plots, box plots, and correlation matrices"
    
    def execute(self, query: str, **kwargs) -> str:
        """Execute visualization recommendations."""
        try:
            self.validate_dataset_requirement()
            
            # Load dataset from storage
            with default_storage.open(self.dataset.processed_file_path, 'rb') as f:
                df = pd.read_parquet(f)
            
            # Get column information
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            date_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
            
            if not numeric_columns and not categorical_columns:
                return "No suitable columns found for visualization."
            
            # Generate visualization recommendations
            response = f"📊 **Visualization Recommendations**\n\n"
            
            # Numeric visualizations
            if numeric_columns:
                response += f"**For Numeric Columns ({numeric_columns[:3]}):**\n"
                if len(numeric_columns) == 1:
                    response += f"- Histogram to show distribution of '{numeric_columns[0]}'\n"
                    response += f"- Box plot to identify outliers\n"
                    response += f"- Density plot for smooth distribution\n"
                elif len(numeric_columns) >= 2:
                    response += f"- Scatter plot matrix to show relationships\n"
                    response += f"- Correlation heatmap\n"
                    response += f"- Pair plot for comprehensive view\n"
                    response += f"- Individual histograms for each column\n"
            
            # Categorical visualizations
            if categorical_columns:
                response += f"\n**For Categorical Columns ({categorical_columns[:3]}):**\n"
                response += f"- Bar charts for value counts\n"
                response += f"- Pie charts for proportions\n"
                response += f"- Count plots for frequency analysis\n"
            
            # Mixed visualizations
            if numeric_columns and categorical_columns:
                response += f"\n**For Mixed Analysis:**\n"
                response += f"- Box plots by category\n"
                response += f"- Violin plots for distribution comparison\n"
                response += f"- Strip plots for individual points\n"
            
            # Time series visualizations
            if date_columns and numeric_columns:
                response += f"\n**For Time Series Analysis:**\n"
                response += f"- Line plots for trends over time\n"
                response += f"- Seasonal decomposition plots\n"
                response += f"- Moving average plots\n"
            
            # Add specific recommendations based on query
            response += self._get_query_specific_recommendations(query, df, numeric_columns, categorical_columns)
            
            response += f"\n*Note: Actual visualizations would be generated and displayed in the UI.*"
            
            self.log_execution(query, response, success=True)
            return response
            
        except Exception as e:
            self.log_error(e)
            error_msg = f"Error creating visualizations: {str(e)}"
            self.log_execution(query, error_msg, success=False)
            return error_msg
    
    def _get_query_specific_recommendations(self, query: str, df: pd.DataFrame, 
                                          numeric_columns: List[str], categorical_columns: List[str]) -> str:
        """Get specific visualization recommendations based on query."""
        query_lower = query.lower()
        recommendations = "\n**Specific Recommendations:**\n"
        
        # Distribution analysis
        if any(word in query_lower for word in ['distribution', 'histogram', 'frequency']):
            if numeric_columns:
                recommendations += f"- Create histograms for: {', '.join(numeric_columns[:3])}\n"
            if categorical_columns:
                recommendations += f"- Create bar charts for: {', '.join(categorical_columns[:3])}\n"
        
        # Relationship analysis
        elif any(word in query_lower for word in ['relationship', 'correlation', 'scatter']):
            if len(numeric_columns) >= 2:
                recommendations += f"- Create scatter plots between: {numeric_columns[0]} and {numeric_columns[1]}\n"
                recommendations += f"- Create correlation heatmap for all numeric variables\n"
        
        # Comparison analysis
        elif any(word in query_lower for word in ['compare', 'comparison', 'group']):
            if categorical_columns and numeric_columns:
                recommendations += f"- Create box plots comparing {numeric_columns[0]} by {categorical_columns[0]}\n"
                recommendations += f"- Create grouped bar charts for categorical comparisons\n"
        
        # Trend analysis
        elif any(word in query_lower for word in ['trend', 'time', 'over time']):
            date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            if date_cols and numeric_columns:
                recommendations += f"- Create line plots showing {numeric_columns[0]} over {date_cols[0]}\n"
                recommendations += f"- Create trend analysis with moving averages\n"
        
        # Outlier analysis
        elif any(word in query_lower for word in ['outlier', 'anomaly', 'extreme']):
            if numeric_columns:
                recommendations += f"- Create box plots for: {', '.join(numeric_columns[:3])}\n"
                recommendations += f"- Create scatter plots to identify outlier patterns\n"
        
        else:
            recommendations += f"- Start with basic histograms and bar charts\n"
            recommendations += f"- Explore relationships with scatter plots\n"
            recommendations += f"- Use box plots to identify outliers\n"
        
        return recommendations
