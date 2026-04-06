"""
Parameter Validation System for Tools
Provides robust validation for tool parameters without disrupting existing tools.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from enum import Enum
import re
import pandas as pd

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for parameter validation errors."""
    
    def __init__(self, field: str, message: str, value: Any = None):
        self.field = field
        self.message = message
        self.value = value
        super().__init__(f"{field}: {message}")


class ValidationWarning:
    """Represents a non-fatal validation issue."""
    
    def __init__(self, field: str, message: str, severity: str = "info"):
        self.field = field
        self.message = message
        self.severity = severity  # 'info', 'warning', 'error'
    
    def __repr__(self):
        return f"ValidationWarning({self.field}, {self.message}, {self.severity})"


class ParameterValidator:
    """
    Validates tool parameters against their schema definitions.
    
    Non-breaking design:
    - Validation is optional (opt-in via validate_parameters=True)
    - Existing tools continue to work without validation
    - Validation errors have multiple severity levels
    - Warnings are collected, not fatal
    """
    
    # Type mapping for validation
    TYPE_VALIDATORS = {
        'text': 'validate_text',
        'number': 'validate_number',
        'select': 'validate_select',
        'multiselect': 'validate_multiselect',
        'checkbox': 'validate_checkbox',
        'textarea': 'validate_textarea',
        'column_select': 'validate_column_select',
        'numeric_column_select': 'validate_numeric_column_select',
        'categorical_column_select': 'validate_categorical_column_select',
        'date_column_select': 'validate_date_column_select',
        'chart_type_select': 'validate_chart_type_select',
        'color_picker': 'validate_color_picker',
        'range_slider': 'validate_range_slider',
    }
    
    def __init__(self, tool_schema, dataset: Optional[pd.DataFrame] = None):
        """
        Initialize validator with a tool's parameter schema.
        
        Args:
            tool_schema: ToolParameterSet instance defining tool parameters
            dataset: Optional pandas DataFrame for column validation
        """
        self.tool_schema = tool_schema
        self.dataset = dataset
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationWarning] = []
    
    def validate(
        self,
        parameters: Dict[str, Any],
        strict: bool = False,
        collect_all: bool = True
    ) -> Tuple[bool, List[ValidationError], List[ValidationWarning]]:
        """
        Validate parameters against the tool schema.
        
        Args:
            parameters: Dictionary of parameter name -> value
            strict: If True, warnings are treated as errors
            collect_all: If True, collect all errors before raising; otherwise fail-fast
        
        Returns:
            Tuple of (is_valid, errors, warnings)
        
        Raises:
            ValidationError: If not collect_all or if errors in strict mode
        """
        self.errors = []
        self.warnings = []
        
        # Check required parameters
        for param in self.tool_schema.get_required_parameters():
            if param.name not in parameters or parameters[param.name] is None:
                error = ValidationError(
                    param.name,
                    f"Required parameter is missing (required=True)",
                    None
                )
                self.errors.append(error)
                if not collect_all:
                    raise error
        
        # Validate provided parameters
        for param_name, param_value in parameters.items():
            param_def = self.tool_schema.get_parameter(param_name)
            
            if param_def is None:
                warning = ValidationWarning(
                    param_name,
                    f"Unknown parameter (not in schema)",
                    "warning"
                )
                self.warnings.append(warning)
                if not collect_all:
                    raise ValidationError(param_name, f"Unknown parameter")
                continue
            
            # Skip None values for optional parameters
            if param_value is None and not param_def.required:
                continue
            
            # Type-specific validation
            try:
                self._validate_parameter_type(param_def, param_value)
            except ValidationError as e:
                self.errors.append(e)
                if not collect_all:
                    raise
            
            # Custom validation rules if defined
            if param_def.validation_rules:
                try:
                    self._apply_validation_rules(param_def, param_value)
                except ValidationError as e:
                    self.errors.append(e)
                    if not collect_all:
                        raise
        
        is_valid = len(self.errors) == 0
        
        if strict and self.warnings:
            # Convert warnings to errors in strict mode
            for warning in self.warnings:
                self.errors.append(ValidationError(warning.field, warning.message))
            is_valid = False
        
        if self.errors and not collect_all:
            raise self.errors[0]
        
        return is_valid, self.errors, self.warnings
    
    def _validate_parameter_type(self, param_def, value: Any) -> None:
        """Dispatch to type-specific validator."""
        param_type = param_def.parameter_type.value
        validator_method = self.TYPE_VALIDATORS.get(param_type)
        
        if not validator_method:
            logger.warning(f"Unknown parameter type: {param_type}, skipping validation")
            return
        
        method = getattr(self, validator_method, None)
        if method:
            method(param_def, value)
        else:
            logger.warning(f"Validator method not found: {validator_method}")
    
    def _apply_validation_rules(self, param_def, value: Any) -> None:
        """Apply custom validation rules."""
        rules = param_def.validation_rules
        
        # Min/max for numbers
        if 'min' in rules and isinstance(value, (int, float)):
            if value < rules['min']:
                raise ValidationError(
                    param_def.name,
                    f"Value {value} is less than minimum {rules['min']}",
                    value
                )
        
        if 'max' in rules and isinstance(value, (int, float)):
            if value > rules['max']:
                raise ValidationError(
                    param_def.name,
                    f"Value {value} exceeds maximum {rules['max']}",
                    value
                )
        
        # Length constraints for strings
        if 'min_length' in rules and isinstance(value, str):
            if len(value) < rules['min_length']:
                raise ValidationError(
                    param_def.name,
                    f"Length {len(value)} is less than minimum {rules['min_length']}",
                    value
                )
        
        if 'max_length' in rules and isinstance(value, str):
            if len(value) > rules['max_length']:
                raise ValidationError(
                    param_def.name,
                    f"Length {len(value)} exceeds maximum {rules['max_length']}",
                    value
                )
        
        # Pattern matching for strings
        if 'pattern' in rules and isinstance(value, str):
            if not re.match(rules['pattern'], value):
                raise ValidationError(
                    param_def.name,
                    f"Value does not match required pattern: {rules['pattern']}",
                    value
                )
        
        # Custom validator function
        if 'custom_validator' in rules and callable(rules['custom_validator']):
            try:
                rules['custom_validator'](value)
            except Exception as e:
                raise ValidationError(
                    param_def.name,
                    f"Custom validation failed: {str(e)}",
                    value
                )
    
    # Type-specific validators
    
    def validate_text(self, param_def, value: Any) -> None:
        """Validate text parameter."""
        if not isinstance(value, str):
            raise ValidationError(
                param_def.name,
                f"Expected string, got {type(value).__name__}",
                value
            )
        if not value.strip():
            raise ValidationError(
                param_def.name,
                "Text cannot be empty or whitespace",
                value
            )
    
    def validate_number(self, param_def, value: Any) -> None:
        """Validate numeric parameter."""
        if not isinstance(value, (int, float)):
            try:
                float(value)  # Try to convert
            except (TypeError, ValueError):
                raise ValidationError(
                    param_def.name,
                    f"Expected number, got {type(value).__name__}",
                    value
                )
    
    def validate_select(self, param_def, value: Any) -> None:
        """Validate select parameter."""
        if param_def.options:
            valid_values = [opt.get('value') for opt in param_def.options]
            if value not in valid_values:
                raise ValidationError(
                    param_def.name,
                    f"Value '{value}' not in allowed options: {valid_values}",
                    value
                )
    
    def validate_multiselect(self, param_def, value: Any) -> None:
        """Validate multiselect parameter."""
        # Can be list or comma-separated string
        if isinstance(value, str):
            values = [v.strip() for v in value.split(',')]
        elif isinstance(value, list):
            values = value
        else:
            raise ValidationError(
                param_def.name,
                f"Expected list or string, got {type(value).__name__}",
                value
            )
        
        if not values:
            raise ValidationError(
                param_def.name,
                "At least one value must be selected",
                value
            )
        
        # Validate against options if defined
        if param_def.options:
            valid_values = [opt.get('value') for opt in param_def.options]
            for v in values:
                if v not in valid_values:
                    raise ValidationError(
                        param_def.name,
                        f"Value '{v}' not in allowed options: {valid_values}",
                        value
                    )
    
    def validate_checkbox(self, param_def, value: Any) -> None:
        """Validate checkbox parameter."""
        if not isinstance(value, (bool, str, int)):
            raise ValidationError(
                param_def.name,
                f"Expected bool or string, got {type(value).__name__}",
                value
            )
        # String variants: 'true', 'false', 'on', 'off', '1', '0'
        if isinstance(value, str):
            if value.lower() not in ('true', 'false', 'on', 'off', '1', '0', 'yes', 'no'):
                raise ValidationError(
                    param_def.name,
                    f"Invalid boolean value: {value}",
                    value
                )
    
    def validate_textarea(self, param_def, value: Any) -> None:
        """Validate textarea parameter."""
        if not isinstance(value, str):
            raise ValidationError(
                param_def.name,
                f"Expected string, got {type(value).__name__}",
                value
            )
    
    def validate_column_select(self, param_def, value: Any) -> None:
        """Validate column selection."""
        self._validate_column_exists(param_def.name, value)
    
    def validate_numeric_column_select(self, param_def, value: Any) -> None:
        """Validate numeric column selection."""
        col = self._validate_column_exists(param_def.name, value)
        if col is not None and self.dataset is not None:
            if not pd.api.types.is_numeric_dtype(self.dataset[col]):
                raise ValidationError(
                    param_def.name,
                    f"Column '{col}' is not numeric",
                    value
                )
    
    def validate_categorical_column_select(self, param_def, value: Any) -> None:
        """Validate categorical column selection."""
        col = self._validate_column_exists(param_def.name, value)
        if col is not None and self.dataset is not None:
            # Check if column is categorical (object type or low cardinality)
            if not (pd.api.types.is_object_dtype(self.dataset[col]) or
                    self.dataset[col].nunique() < 50):
                self.warnings.append(ValidationWarning(
                    param_def.name,
                    f"Column '{col}' may not be categorical (high cardinality)",
                    "warning"
                ))
    
    def validate_date_column_select(self, param_def, value: Any) -> None:
        """Validate date column selection."""
        col = self._validate_column_exists(param_def.name, value)
        if col is not None and self.dataset is not None:
            if not pd.api.types.is_datetime64_any_dtype(self.dataset[col]):
                self.warnings.append(ValidationWarning(
                    param_def.name,
                    f"Column '{col}' is not datetime type",
                    "warning"
                ))
    
    def validate_chart_type_select(self, param_def, value: Any) -> None:
        """Validate chart type selection."""
        valid_types = ['histogram', 'scatter', 'box', 'line', 'bar', 'pie', 'heatmap']
        if value not in valid_types:
            raise ValidationError(
                param_def.name,
                f"Invalid chart type '{value}'. Valid: {valid_types}",
                value
            )
    
    def validate_color_picker(self, param_def, value: Any) -> None:
        """Validate color picker value."""
        if not isinstance(value, str):
            raise ValidationError(
                param_def.name,
                f"Expected hex color string, got {type(value).__name__}",
                value
            )
        # Simple hex color validation
        if not re.match(r'^#[0-9a-fA-F]{6}$', value):
            raise ValidationError(
                param_def.name,
                f"Invalid hex color format: {value}",
                value
            )
    
    def validate_range_slider(self, param_def, value: Any) -> None:
        """Validate range slider value."""
        if isinstance(value, dict):
            if 'min' not in value or 'max' not in value:
                raise ValidationError(
                    param_def.name,
                    "Range must contain 'min' and 'max' keys",
                    value
                )
            try:
                min_val = float(value['min'])
                max_val = float(value['max'])
                if min_val > max_val:
                    raise ValidationError(
                        param_def.name,
                        f"Min ({min_val}) cannot be greater than max ({max_val})",
                        value
                    )
            except (TypeError, ValueError) as e:
                raise ValidationError(
                    param_def.name,
                    f"Range values must be numeric: {str(e)}",
                    value
                )
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            try:
                min_val, max_val = float(value[0]), float(value[1])
                if min_val > max_val:
                    raise ValidationError(
                        param_def.name,
                        f"Min ({min_val}) cannot be greater than max ({max_val})",
                        value
                    )
            except (TypeError, ValueError) as e:
                raise ValidationError(
                    param_def.name,
                    f"Range values must be numeric: {str(e)}",
                    value
                )
        else:
            raise ValidationError(
                param_def.name,
                f"Expected dict with min/max or 2-tuple, got {type(value).__name__}",
                value
            )
    
    def _validate_column_exists(self, param_name: str, value: Any) -> Optional[str]:
        """Helper to validate column exists in dataset."""
        if not isinstance(value, str):
            raise ValidationError(
                param_name,
                f"Expected column name (string), got {type(value).__name__}",
                value
            )
        
        if self.dataset is not None:
            if value not in self.dataset.columns:
                raise ValidationError(
                    param_name,
                    f"Column '{value}' not found in dataset. Available: {list(self.dataset.columns)}",
                    value
                )
        
        return value


def create_validator(tool_schema, dataset: Optional[pd.DataFrame] = None) -> ParameterValidator:
    """Factory function for creating validators."""
    return ParameterValidator(tool_schema, dataset)
