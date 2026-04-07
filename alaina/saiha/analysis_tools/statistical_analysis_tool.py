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
        return "Provides a combined numeric overview: descriptive statistics (count, mean, std, min/max, quartiles) AND a basic Pearson correlation table for all numeric columns. Use for a quick statistical snapshot. For a dedicated correlation heatmap use correlation_matrix; for detailed skewness/kurtosis use descriptive_statistics."
    
    def get_parameters_schema(self) -> ToolParameterSet:
        from .tool_parameters import ToolParameterSet, ToolParameter, ParameterType
        params = ToolParameterSet(tool_name=self.name)
        params.add_parameter(ToolParameter(
            name="columns",
            parameter_type=ParameterType.MULTISELECT,
            label="Select Columns",
            description="Numeric columns to include in the statistical analysis. If empty, all numeric columns will be used.",
            required=False,
            column_source="numeric"
        ))
        return params

    def execute(self, query: str = "", **kwargs: Any) -> dict:
        """Execute statistical analysis."""
        try:
            self.validate_dataset_requirement()
            
            # Use efficient 'Pass-by-Memory' loading (Bug 12)
            df = self.load_dataset()
            
            # Get numeric columns (either from params or auto-discovery)
            target_cols = kwargs.get("columns", [])
            if not target_cols:
                target_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            else:
                if isinstance(target_cols, str):
                    target_cols = [target_cols]
                # Filter to only valid numeric columns
                target_cols = [c for c in target_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
            
            if not target_cols:
                return {"status": "error", "summary": "No numeric columns found/selected for statistical analysis."}
            
            # Perform basic statistical analysis
            stats_results = {}
            
            # Descriptive statistics
            desc_stats = df[target_cols].describe()
            stats_results['descriptive'] = desc_stats.to_dict()
            
            # Correlation analysis if multiple numeric columns
            corr_matrix = None
            if len(target_cols) > 1:
                corr_matrix = df[target_cols].corr()
                stats_results['correlation'] = corr_matrix.to_dict()
            
            # Standardize on 'artifacts' (Bug 13)
            artifacts = [
                {
                    'type': 'table', 'title': 'Descriptive Statistics Overview',
                    'headers': ['Metric'] + desc_stats.columns.tolist(),
                    'data': [[idx] + [str(v) if not pd.isna(v) else "N/A" for v in row] for idx, row in desc_stats.iterrows()]
                }
            ]
            
            if corr_matrix is not None:
                artifacts.append({
                    'type': 'table', 'title': 'Correlation Matrix (Pearson)',
                    'headers': ['Variable'] + corr_matrix.columns.tolist(),
                    'data': [[idx] + [f"{v:.4f}" if not pd.isna(v) else "1.0000" for v in row] for idx, row in corr_matrix.iterrows()]
                })

            # --- KEY FIX: Link the Interpretation Logic (Bug 12.5.1) ---
            interpretation = self._add_interpretation(desc_stats, corr_matrix)
            summary = f"Statistical analysis complete for {len(target_cols)} numeric columns. " + interpretation.replace("**", "").replace("\n", " ").strip()
            
            return {
                "status": "ok",
                "summary": summary,
                "artifacts": artifacts,
                "data": stats_results, # Keep for backward compatibility
                "meta": {"tool_name": self.name, "parameters": kwargs}
            }
            
        except Exception as e:
            self.log_error(e)
            return {"status": "error", "summary": f"Error performing statistical analysis: {str(e)}"}
    
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
