"""
Tool Parameter System
Defines parameter types and validation for dynamic form generation.
"""

from enum import Enum
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
import logging
import re

logger = logging.getLogger(__name__)


class ParameterType(Enum):
    """Types of parameters that tools can accept."""
    TEXT = "text"
    NUMBER = "number"
    SELECT = "select"
    MULTISELECT = "multiselect"
    CHECKBOX = "checkbox"
    TEXTAREA = "textarea"
    COLUMN_SELECT = "column_select"
    NUMERIC_COLUMN_SELECT = "numeric_column_select"
    CATEGORICAL_COLUMN_SELECT = "categorical_column_select"
    DATE_COLUMN_SELECT = "date_column_select"
    CHART_TYPE_SELECT = "chart_type_select"
    COLOR_PICKER = "color_picker"
    RANGE_SLIDER = "range_slider"


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    parameter_type: ParameterType
    label: str
    description: str = ""
    required: bool = True
    default_value: Any = None
    options: List[Dict[str, Any]] = field(default_factory=list)
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    placeholder: str = ""
    help_text: str = ""
    column_source: Optional[str] = None  # 'numeric', 'categorical', or 'date'


@dataclass
class ToolParameterSet:
    """Set of parameters for a tool."""
    tool_name: str
    parameters: List[ToolParameter] = field(default_factory=list)
    
    def add_parameter(self, parameter: ToolParameter):
        """Add a parameter to the set."""
        self.parameters.append(parameter)
    
    def get_parameter(self, name: str) -> Optional[ToolParameter]:
        """Get a parameter by name."""
        for param in self.parameters:
            if param.name == name:
                return param
        return None
    
    def get_required_parameters(self) -> List[ToolParameter]:
        """Get all required parameters."""
        return [param for param in self.parameters if param.required]
    
    def get_optional_parameters(self) -> List[ToolParameter]:
        """Get all optional parameters."""
        return [param for param in self.parameters if not param.required]


class ToolParameterRegistry:
    """Registry for tool parameter definitions."""
    
    def __init__(self):
        self._parameter_sets: Dict[str, ToolParameterSet] = {}
        self._initialize_default_parameters()
    
    def register_parameters(self, tool_name: str, parameter_set: ToolParameterSet):
        """Register parameters for a tool."""
        self._parameter_sets[tool_name] = parameter_set
        logger.info(f"Registered parameters for tool: {tool_name}")
    
    def get_parameters(self, tool_name: str) -> Optional[ToolParameterSet]:
        """Get parameters for a tool."""
        return self._parameter_sets.get(tool_name)
    
    def get_all_parameter_sets(self) -> Dict[str, ToolParameterSet]:
        """Get all parameter sets."""
        return self._parameter_sets.copy()
    
    def _initialize_default_parameters(self):
        """Initialize default parameter sets for built-in tools."""
        
        # Dataset Overview Tool
        dataset_overview_params = ToolParameterSet("dataset_overview")
        dataset_overview_params.add_parameter(ToolParameter(
            name="include_missing_analysis",
            parameter_type=ParameterType.CHECKBOX,
            label="Include Missing Data Analysis",
            description="Include detailed missing data analysis in the overview",
            required=False,
            default_value=True
        ))
        self.register_parameters("dataset_overview", dataset_overview_params)
        
        # Column Analysis Tool
        column_analysis_params = ToolParameterSet("column_analysis")
        column_analysis_params.add_parameter(ToolParameter(
            name="columns",
            parameter_type=ParameterType.MULTISELECT,
            label="Select Columns",
            description="Choose columns to analyze",
            required=True,
            options=[],  # Will be populated dynamically
            help_text="Select one or more columns to analyze"
        ))
        column_analysis_params.add_parameter(ToolParameter(
            name="include_distribution",
            parameter_type=ParameterType.CHECKBOX,
            label="Include Distribution Analysis",
            description="Include distribution analysis for numeric columns",
            required=False,
            default_value=True
        ))
        self.register_parameters("column_analysis", column_analysis_params)
        
        # Data Quality Tool
        data_quality_params = ToolParameterSet("data_quality")
        data_quality_params.add_parameter(ToolParameter(
            name="check_outliers",
            parameter_type=ParameterType.CHECKBOX,
            label="Check for Outliers",
            description="Include outlier detection in quality assessment",
            required=False,
            default_value=True
        ))
        data_quality_params.add_parameter(ToolParameter(
            name="check_duplicates",
            parameter_type=ParameterType.CHECKBOX,
            label="Check for Duplicates",
            description="Include duplicate detection in quality assessment",
            required=False,
            default_value=True
        ))
        self.register_parameters("data_quality", data_quality_params)
        
        # Visualization Tool
        visualization_params = ToolParameterSet("visualization")
        visualization_params.add_parameter(ToolParameter(
            name="chart_type",
            parameter_type=ParameterType.CHART_TYPE_SELECT,
            label="Chart Type",
            description="Select the type of visualization",
            required=True,
            options=[
                {"value": "histogram", "label": "Histogram", "icon": "fas fa-chart-bar"},
                {"value": "scatter", "label": "Scatter Plot", "icon": "fas fa-chart-scatter"},
                {"value": "box", "label": "Box Plot", "icon": "fas fa-chart-box"},
                {"value": "line", "label": "Line Chart", "icon": "fas fa-chart-line"},
                {"value": "bar", "label": "Bar Chart", "icon": "fas fa-chart-bar"},
                {"value": "pie", "label": "Pie Chart", "icon": "fas fa-chart-pie"},
                {"value": "heatmap", "label": "Heatmap", "icon": "fas fa-th"}
            ],
            default_value="histogram"
        ))
        visualization_params.add_parameter(ToolParameter(
            name="x_column",
            parameter_type=ParameterType.COLUMN_SELECT,
            label="X-Axis Column",
            description="Column for X-axis",
            required=True,
            options=[]  # Will be populated dynamically
        ))
        visualization_params.add_parameter(ToolParameter(
            name="y_column",
            parameter_type=ParameterType.COLUMN_SELECT,
            label="Y-Axis Column",
            description="Column for Y-axis (optional for some chart types)",
            required=False,
            options=[]  # Will be populated dynamically
        ))
        visualization_params.add_parameter(ToolParameter(
            name="color_column",
            parameter_type=ParameterType.CATEGORICAL_COLUMN_SELECT,
            label="Color Column",
            description="Column for color grouping (optional)",
            required=False,
            options=[]  # Will be populated dynamically
        ))
        visualization_params.add_parameter(ToolParameter(
            name="chart_title",
            parameter_type=ParameterType.TEXT,
            label="Chart Title",
            description="Title for the chart",
            required=False,
            placeholder="Enter chart title"
        ))
        self.register_parameters("visualization", visualization_params)


# Global parameter registry instance
parameter_registry = ToolParameterRegistry()


def get_tool_parameters(tool_name: str) -> Optional[ToolParameterSet]:
    """Get parameters for a tool."""
    return parameter_registry.get_parameters(tool_name)


def register_tool_parameters(tool_name: str, parameter_set: ToolParameterSet):
    """Register parameters for a tool."""
    parameter_registry.register_parameters(tool_name, parameter_set)


@dataclass
class ValidationError:
    """Represents a validation error for a tool parameter."""
    parameter: str
    message: str
    details: Dict[str, Any]


@dataclass
class ValidationResult:
    """Result of parameter validation."""
    is_valid: bool
    errors: List[str]
    suggestions: List[str]
    details: Dict[str, Any]


class ToolParameterValidator:
    """Validates tool parameters against their specifications."""
    
    def __init__(self, tool_spec: Dict[str, Any]):
        """Initialize validator with tool specification."""
        self.parameters = tool_spec
        
    def validate(self, params: Dict[str, Any]) -> ValidationResult:
        """
        Validate parameters against tool specification.
        
        Args:
            params: Parameters to validate
            
        Returns:
            ValidationResult with validation details
        """
        errors = []
        suggestions = []
        details = {}
        
        # Check required parameters
        for param_name, spec in self.parameters.items():
            if spec.get("required", False) and param_name not in params:
                errors.append(f"Missing required parameter {param_name}")  # Make sure "missing" appears in the error
                suggestions.append(f"Add {param_name} parameter")
                
        # Validate parameter types
        for param_name, value in params.items():
            if param_name in self.parameters:
                spec = self.parameters[param_name]
                if not self._validate_type(value, spec["type"]):
                    errors.append(f"Invalid type for {param_name}")
                    suggestions.append(
                        f"Parameter {param_name} should be of type {spec['type']}"
                    )
                    
        # Check constraints if any
        for param_name, value in params.items():
            if param_name in self.parameters:
                spec = self.parameters[param_name]
                constraint_errors = self._check_constraints(param_name, value, spec)
                errors.extend(constraint_errors)
                
        # Add details about validated parameters
        details["validated_parameters"] = list(params.keys())
        details["missing_parameters"] = [
            p for p in self.parameters 
            if p not in params and self.parameters[p].get("required", False)
        ]
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            suggestions=suggestions,
            details=details
        )
        
    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """Validate parameter type."""
        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict
        }
        
        if expected_type not in type_map:
            return True  # Skip validation for unknown types
            
        expected_types = type_map[expected_type]
        return isinstance(value, expected_types)
        
    def _check_constraints(
        self, 
        param_name: str, 
        value: Any, 
        spec: Dict[str, Any]
    ) -> List[str]:
        """Check parameter constraints."""
        errors = []
        
        # Check numeric constraints
        if isinstance(value, (int, float)):
            if "minimum" in spec and value < spec["minimum"]:
                errors.append(
                    f"Parameter {param_name} must be >= {spec['minimum']}"
                )
            if "maximum" in spec and value > spec["maximum"]:
                errors.append(
                    f"Parameter {param_name} must be <= {spec['maximum']}"
                )
                
        # Check string constraints
        if isinstance(value, str):
            if "minLength" in spec and len(value) < spec["minLength"]:
                errors.append(
                    f"Parameter {param_name} must be at least {spec['minLength']} characters"
                )
            if "maxLength" in spec and len(value) > spec["maxLength"]:
                errors.append(
                    f"Parameter {param_name} must be at most {spec['maxLength']} characters"
                )
            if "pattern" in spec and not re.match(spec["pattern"], value):
                errors.append(
                    f"Parameter {param_name} must match pattern {spec['pattern']}"
                )
                
        # Check array constraints
        if isinstance(value, list):
            if "minItems" in spec and len(value) < spec["minItems"]:
                errors.append(
                    f"Parameter {param_name} must have at least {spec['minItems']} items"
                )
            if "maxItems" in spec and len(value) > spec["maxItems"]:
                errors.append(
                    f"Parameter {param_name} must have at most {spec['maxItems']} items"
                )
                
        return errors