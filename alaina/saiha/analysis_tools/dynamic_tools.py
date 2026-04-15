import logging
from typing import List, Dict, Any, Type, Optional
from langchain_core.tools import StructuredTool
from .tool_registry import ToolRegistry
from .base_tool import BaseAnalysisTool
from .tool_parameters import ToolParameter, ParameterType
from saiha.models import AnalysisSession, Dataset
from django.contrib.auth.models import User
from pydantic import create_model, Field, BaseModel

logger = logging.getLogger(__name__)

# Import the actual initialized registry instance from the package
from . import tool_registry as registry_instance

def map_param_to_pytype(param: ToolParameter) -> Any:
    """Maps internal ParameterType to Python types for Pydantic."""
    ptype = param.parameter_type
    
    if ptype in [ParameterType.TEXT, ParameterType.TEXTAREA, ParameterType.COLOR_PICKER,
                 ParameterType.SELECT, ParameterType.CHART_TYPE_SELECT, ParameterType.COLUMN_SELECT, 
                 ParameterType.NUMERIC_COLUMN_SELECT, ParameterType.CATEGORICAL_COLUMN_SELECT, 
                 ParameterType.DATE_COLUMN_SELECT]:
        return str
    elif ptype == ParameterType.NUMBER:
        return float
    elif ptype == ParameterType.CHECKBOX:
        return bool
    elif ptype in [ParameterType.MULTISELECT]:
        return List[str]
    elif ptype == ParameterType.RANGE_SLIDER:
        return float
    else:
        return str # Fallback

def create_langchain_tool(tool_instance: BaseAnalysisTool, user_id: int, session_id: str, dataset_id: str, final_name: str) -> StructuredTool:
    """Wraps a BaseAnalysisTool into a LangChain StructuredTool with explicit pydantic v2 schema."""
    
    param_set = tool_instance.get_parameters_schema()
    fields = {}
    
    # Add 'query' as a mandatory keyword for the model
    fields['query'] = (str, Field(default="", description="The analysis context or core question."))

    if param_set and param_set.parameters:
        for param in param_set.parameters:
            pytype = map_param_to_pytype(param)
            description = param.description or param.label
            if pytype == List[str]:
                fields[param.name] = (List[str], Field(default_factory=list, description=description))
            elif param.required:
                fields[param.name] = (pytype, Field(..., description=description))
            else:
                fields[param.name] = (Optional[pytype], Field(default=None, description=description))
    
    # Create the model using unique names to avoid Pydantic caching issues
    import uuid
    safe_model_name = f"Schema_{final_name}_{uuid.uuid4().hex[:8]}"
    ArgsModel = create_model(safe_model_name, __base__=BaseModel, **fields)

    def tool_func(**kwargs):
        try:
            if not hasattr(tool_instance, 'user') or not tool_instance.user:
                tool_instance.user = User.objects.get(id=user_id)
            if not hasattr(tool_instance, 'session') or not tool_instance.session:
                tool_instance.session = AnalysisSession.objects.get(id=session_id)
            if dataset_id and dataset_id != "None" and (not hasattr(tool_instance, 'dataset') or not tool_instance.dataset):
                # Ownership guard: pin Dataset lookup to the session owner
                _session_obj = AnalysisSession.objects.get(id=session_id)
                tool_instance.dataset = Dataset.objects.get(id=dataset_id, user=_session_obj.user)
            
            query = kwargs.pop('query', f"Run {tool_instance.name}")
            result = tool_instance.run(query=query, **kwargs)
            import json
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Error executing tool {final_name}: {e}")
            import json
            return json.dumps({"success": False, "error": str(e)})

    return StructuredTool(
        name=final_name,
        description=tool_instance.description,
        func=tool_func,
        args_schema=ArgsModel
    )

def get_dynamic_tools(user_id: int, session_id: str, dataset_id: str) -> List[StructuredTool]:
    """Returns a list of all tools in the registry wrapped for LangChain, with class-deduplication."""
    dynamic_tools = []
    
    # Import from __init__ to ensure we have the fully initialized global registry
    from . import tool_registry as global_registry_instance
    all_tools_dict = global_registry_instance.get_all_tools()
    
    excluded_names = ['recommendations', 'check_dataset_schema', 'get_current_time', 'data_quality_assessment'] 
    seen_classes = set()
    seen_names = set()

    for tool_name, tool_inst in all_tools_dict.items():
        # normalize name
        clean_name = tool_name.lower().replace(" ", "_").replace("-", "_")
        tool_class = tool_inst.__class__
        
        # Deduplicate strictly:
        # 1. Skip excluded names
        # 2. Skip already seen registry keys
        # 3. Skip duplicate tool logic (class-based)
        if clean_name in excluded_names or clean_name in seen_names or tool_class in seen_classes:
            continue
            
        try:
            lc_tool = create_langchain_tool(tool_inst, user_id, session_id, dataset_id, clean_name)
            dynamic_tools.append(lc_tool)
            seen_names.add(clean_name)
            seen_classes.add(tool_class)
        except Exception as e:
            logger.error(f"Failed to wrap tool {tool_name}: {e}")
            
    return dynamic_tools
