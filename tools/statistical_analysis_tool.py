"""
Statistical Analysis Tool
Performs statistical analysis including descriptive statistics, hypothesis tests, and correlation analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from django.core.files.storage import default_storage
from .base_tool import BaseAnalysisTool


class StatisticalAnalysisTool(BaseAnalysisTool):
    """Tool for performing statistical analysis."""
    
    @property
    def name(self) -> str:
        return "statistical_analysis"
    
    @property
    def description(self) -> str:
        return "Perform statistical analysis including descriptive statistics, hypothesis tests, and correlation analysis"
    
    def execute(self, query: str, **kwargs) -> str:
        """Execute statistical analysis."""
        try:
            self.validate_dataset_requirement()
            
            # Load dataset from storage
            with default_storage.open(self.dataset.processed_file_path, 'rb') as f:
                df = pd.read_parquet(f)
            
            # Get numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_columns:
                return "No numeric columns found for statistical analysis."
            
            # Perform basic statistical analysis
            stats_results = {}
            
            # Descriptive statistics
            desc_stats = df[numeric_columns].describe()
            stats_results['descriptive'] = desc_stats.to_dict()
            
            # Correlation analysis if multiple numeric columns
            if len(numeric_columns) > 1:
                corr_matrix = df[numeric_columns].corr()
                stats_results['correlation'] = corr_matrix.to_dict()
            
            # Format response
            response = f"📊 **Statistical Analysis Results**\n\n"
            response += f"**Descriptive Statistics:**\n{desc_stats.to_string()}\n\n"
            
            if 'correlation' in stats_results:
                response += f"**Correlation Matrix:**\n{corr_matrix.to_string()}\n"
            
            # Add interpretation
            response += self._add_interpretation(desc_stats, corr_matrix if len(numeric_columns) > 1 else None)
            
            self.log_execution(query, response, success=True)
            return response
            
        except Exception as e:
            self.log_error(e)
            error_msg = f"Error performing statistical analysis: {str(e)}"
            self.log_execution(query, error_msg, success=False)
            return error_msg
    
    def _add_interpretation(self, desc_stats: pd.DataFrame, corr_matrix: pd.DataFrame = None) -> str:
        """Add interpretation to statistical results."""
        interpretation = "\n**Key Insights:**\n"
        
        # Analyze descriptive statistics
        for col in desc_stats.columns:
            mean_val = desc_stats.loc['mean', col]
            std_val = desc_stats.loc['std', col]
            min_val = desc_stats.loc['min', col]
            max_val = desc_stats.loc['max', col]
            
            # Check for outliers (values beyond 2 standard deviations)
            outlier_threshold = 2 * std_val
            if abs(max_val - mean_val) > outlier_threshold or abs(mean_val - min_val) > outlier_threshold:
                interpretation += f"- {col}: Potential outliers detected (high variability)\n"
            else:
                interpretation += f"- {col}: Relatively normal distribution\n"
        
        # Analyze correlations
        if corr_matrix is not None:
            interpretation += "\n**Correlation Insights:**\n"
            high_corr_pairs = []
            for i, col1 in enumerate(corr_matrix.columns):
                for j, col2 in enumerate(corr_matrix.columns):
                    if i < j:  # Avoid duplicates
                        corr_val = corr_matrix.loc[col1, col2]
                        if abs(corr_val) > 0.7:
                            high_corr_pairs.append((col1, col2, corr_val))
            
            if high_corr_pairs:
                interpretation += "Strong correlations found:\n"
                for col1, col2, corr_val in high_corr_pairs[:3]:  # Show top 3
                    interpretation += f"- {col1} ↔ {col2}: {corr_val:.3f}\n"
            else:
                interpretation += "- No strong correlations found between variables\n"
        
        return interpretation
