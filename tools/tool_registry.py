"""
Tool Registry for managing analysis tools with database integration.
"""

import logging
from typing import Dict, List, Any, Optional, Type
from django.db import models

from quantalytics.models import Tool, ToolCategory
from .base_tool import BaseAnalysisTool

logger = logging.getLogger(__name__)


from .data_type_conversion_tool import DataTypeConversionTool
from .sample_size_estimator_tool import SampleSizeEstimatorTool
from .effect_size_calculator_tool import EffectSizeCalculatorTool
from .randomization_generator_tool import RandomizationGeneratorTool
from .precision_analysis_tool import PrecisionAnalysisTool
from .outlier_detection_tool import OutlierDetectionTool
from .outlier_treatment_tool import OutlierTreatmentTool
from .linear_regression_tool import LinearRegressionTool
from .logistic_regression_tool import LogisticRegressionTool
from .one_way_anova_tool import OneWayAnovaTool
from .two_way_anova_tool import TwoWayAnovaTool
from .one_sample_t_test_tool import OneSampleTTestTool
from .two_sample_t_test_tool import TwoSampleTTestTool
from .paired_t_test_tool import PairedTTestTool
from .descriptive_statistics_tool import DescriptiveStatisticsTool
from .correlation_matrix_tool import CorrelationMatrixTool
from .histogram_tool import HistogramTool
from .bar_chart_tool import BarChartTool
from .line_chart_tool import LineChartTool
from .scatter_plot_tool import ScatterPlotTool
from .box_plot_tool import BoxPlotTool
from .pca_tool import PCATool
from .kmeans_clustering_tool import KMeansClusteringTool
from .decision_tree_tool import DecisionTreeTool
from .factor_analysis_tool import FactorAnalysisTool
from .data_quality_tool import DataQualityTool
from .dataset_overview_tool import DatasetOverviewTool

class ToolRegistry:
    """Registry for managing analysis tools with database integration."""
    
    def __init__(self):
        """Initialize the tool registry."""
        self._tools: Dict[str, Type[BaseAnalysisTool]] = {}
        self._db_tools_cache: Dict[int, Tool] = {}
        self._categories_cache: Dict[int, ToolCategory] = {}
        
        # Register core tools
        self.register_tool('data_type_conversion', DataTypeConversionTool)
        self.register_tool('sample_size_estimator', SampleSizeEstimatorTool)
        self.register_tool('effect_size_calculator', EffectSizeCalculatorTool)
        self.register_tool('randomization_generator', RandomizationGeneratorTool)
        self.register_tool('precision_analysis', PrecisionAnalysisTool)
        self.register_tool('outlier_detection', OutlierDetectionTool)
        self.register_tool('outlier_treatment', OutlierTreatmentTool)
        self.register_tool('linear_regression', LinearRegressionTool)
        self.register_tool('logistic_regression', LogisticRegressionTool)
        self.register_tool('one_way_anova', OneWayAnovaTool)
        self.register_tool('two_way_anova', TwoWayAnovaTool)
        self.register_tool('one_sample_t_test', OneSampleTTestTool)
        self.register_tool('two_sample_t_test', TwoSampleTTestTool)
        self.register_tool('paired_t_test', PairedTTestTool)
        self.register_tool('descriptive_statistics', DescriptiveStatisticsTool)
        self.register_tool('correlation_matrix', CorrelationMatrixTool)
        self.register_tool('histogram', HistogramTool)
        self.register_tool('bar_chart', BarChartTool)
        self.register_tool('line_chart', LineChartTool)
        self.register_tool('scatter_plot', ScatterPlotTool)
        self.register_tool('box_plot', BoxPlotTool)
        self.register_tool('pca', PCATool)
        self.register_tool('kmeans_clustering', KMeansClusteringTool)
        self.register_tool('decision_tree', DecisionTreeTool)
        self.register_tool('factor_analysis', FactorAnalysisTool)
        self.register_tool('data_quality', DataQualityTool)
        self.register_tool('dataset_overview', DatasetOverviewTool)
    
    def register_tool(self, tool_name: str, tool_class: Type[BaseAnalysisTool]):
        """
        Register a tool class.
        
        Args:
           tool_name: Unique tool identifier
           tool_class: Tool class that extends BaseAnalysisTool
        """
        self._tools[tool_name] = tool_class
        logger.info(f"Registered tool: {tool_name}")
    
    def get_tool(self, tool_name: str, agent=None, tool_id: int = None) -> Optional[BaseAnalysisTool]:
        """
        Get a tool instance by name or ID.
        
        Args:
            tool_name: Tool name
            agent: DataAnalysisAgent instance
            tool_id: Database tool ID (optional)
            
        Returns:
            Tool instance or None
        """
        try:
            tool_class = self._tools.get(tool_name)
            if not tool_class:
                logger.warning(f"Tool not found: {tool_name}")
                return None
            
            # Get database tool if ID provided
            db_tool = None
            if tool_id:
                db_tool = self._get_db_tool(tool_id)
            
            # Create tool instance
            tool_instance = tool_class(agent=agent, db_tool=db_tool)
            return tool_instance
            
        except Exception as e:
            logger.error(f"Error getting tool {tool_name}: {e}")
            return None
    
    def get_tool_by_id(self, tool_id: int, agent=None) -> Optional[BaseAnalysisTool]:
        """
        Get a tool instance by database ID.
        
        Args:
            tool_id: Database tool ID
            agent: DataAnalysisAgent instance
            
        Returns:
            Tool instance or None
        """
        try:
            db_tool = self._get_db_tool(tool_id)
            if not db_tool:
                return None
            
            # Find tool class by name (assuming tool name matches registered name)
            tool_name = db_tool.name.lower().replace(' ', '_')
            tool_class = self._tools.get(tool_name)
            
            if not tool_class:
                # Try to find by tool type
                tool_class = self._tools.get(db_tool.tool_type)
            
            if not tool_class:
                logger.warning(f"No tool class found for database tool: {db_tool.name}")
                return None
            
            return tool_class(agent=agent, db_tool=db_tool)
            
        except Exception as e:
            logger.error(f"Error getting tool by ID {tool_id}: {e}")
            return None
    
    def get_tools_by_category(self, category_id: int, agent=None) -> List[BaseAnalysisTool]:
        """
        Get all tools for a specific category.
        
        Args:
            category_id: Category ID
            agent: DataAnalysisAgent instance
            
        Returns:
            List of tool instances
        """
        try:
            tools = []
            db_tools = Tool.objects.filter(
                category_id=category_id,
                is_active=True
            ).order_by('order', 'name')
            
            for db_tool in db_tools:
                tool_instance = self.get_tool_by_id(db_tool.id, agent)
                if tool_instance:
                    tools.append(tool_instance)
            
            return tools
            
        except Exception as e:
            logger.error(f"Error getting tools by category {category_id}: {e}")
            return []
    
    def get_all_tools(self, agent=None) -> Dict[str, BaseAnalysisTool]:
        """
        Get all registered tools.
        
        Args:
            agent: DataAnalysisAgent instance
            
        Returns:
            Dictionary of tool instances
        """
        tools = {}
        for tool_name, tool_class in self._tools.items():
            try:
                tool_instance = tool_class(agent=agent)
                tools[tool_name] = tool_instance
            except Exception as e:
                logger.error(f"Error creating tool {tool_name}: {e}")
        
        return tools
    
    def get_active_db_tools(self, agent=None) -> List[BaseAnalysisTool]:
        """
        Get all active tools from database.
        
        Args:
            agent: DataAnalysisAgent instance
            
        Returns:
            List of tool instances
        """
        try:
            tools = []
            db_tools = Tool.objects.filter(is_active=True).order_by('category__order', 'order')
            
            for db_tool in db_tools:
                tool_instance = self.get_tool_by_id(db_tool.id, agent)
                if tool_instance:
                    tools.append(tool_instance)
            
            return tools
            
        except Exception as e:
            logger.error(f"Error getting active database tools: {e}")
            return []
    
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())
    
    def list_db_tools(self) -> List[Dict[str, Any]]:
        """List all database tools with their information."""
        try:
            db_tools = Tool.objects.filter(is_active=True).select_related('category')
            return [
                {
                    'id': tool.id,
                    'name': tool.name,
                    'description': tool.description,
                    'category_id': tool.category.id,
                    'category_name': tool.category.name,
                    'tool_type': tool.tool_type,
                    'api_endpoint': tool.api_endpoint,
                    'requires_dataset': tool.requires_dataset,
                    'order': tool.order
                }
                for tool in db_tools
            ]
        except Exception as e:
            logger.error(f"Error listing database tools: {e}")
            return []
    
    def get_categories(self) -> List[Dict[str, Any]]:
        """Get all tool categories."""
        try:
            categories = ToolCategory.objects.filter(is_active=True).order_by('order')
            return [
                {
                    'id': cat.id,
                    'name': cat.name,
                    'description': cat.description,
                    'icon': cat.icon,
                    'color': cat.color,
                    'order': cat.order
                }
                for cat in categories
            ]
        except Exception as e:
            logger.error(f"Error getting categories: {e}")
            return []
    
    def _get_db_tool(self, tool_id: int) -> Optional[Tool]:
        """Get database tool with caching."""
        if tool_id not in self._db_tools_cache:
            try:
                self._db_tools_cache[tool_id] = Tool.objects.select_related('category').get(id=tool_id)
            except Tool.DoesNotExist:
                logger.warning(f"Database tool not found: {tool_id}")
                return None
        
        return self._db_tools_cache[tool_id]
    
    def clear_cache(self):
        """Clear all caches."""
        self._db_tools_cache.clear()
        self._categories_cache.clear()
        logger.info("Tool registry cache cleared")
    
    def get_tool_stats(self) -> Dict[str, Any]:
        """Get tool registry statistics."""
        return {
            'registered_tools': len(self._tools),
            'cached_db_tools': len(self._db_tools_cache),
            'cached_categories': len(self._categories_cache),
            'active_db_tools': Tool.objects.filter(is_active=True).count(),
            'total_categories': ToolCategory.objects.filter(is_active=True).count()
        }
