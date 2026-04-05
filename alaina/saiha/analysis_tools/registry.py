import os
import importlib
import logging
import inspect
from typing import Dict, List, Type, Optional, Any
from .base_tool import BaseAnalysisTool

logger = logging.getLogger(__name__)

class ToolRegistry:
    """
    Singleton for discovering and initializing hardened tool classes.
    Auto-discovers and registers all 84 tools in the isolated folder.
    """
    _instance = None
    _tools: Dict[str, Type[BaseAnalysisTool]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ToolRegistry, cls).__new__(cls)
            # Discovery is moved to a manual call or delayed initialization
            # to avoid circular imports during module registration.
        return cls._instance

    def _get_tool_name(self, tool_class: Type[BaseAnalysisTool]) -> Optional[str]:
        """Supports both hardened (class attr) and legacy (property/method)."""
        name = getattr(tool_class, 'name', None)
        if isinstance(name, property):
            try:
                return tool_class().name
            except Exception:
                return None
        return name if isinstance(name, str) else None

    def discover_tools(self):
        """Auto-discovers all tool classes in the analysis_tools package."""
        if self._tools: return # Already discovered
        
        base_path = os.path.dirname(__file__)
        for filename in os.listdir(base_path):
            if (filename.endswith(".py") and not filename.startswith("__") 
                and filename not in ["base_tool.py", "registry.py", "tool_parameters.py", "parameter_validator.py"]):
                module_name = f".{filename[:-3]}"
                try:
                    module = importlib.import_module(module_name, package="saiha.analysis_tools")
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and issubclass(obj, BaseAnalysisTool) 
                            and obj is not BaseAnalysisTool):
                            tool_name = self._get_tool_name(obj)
                            if tool_name:
                                self.register(tool_name, obj)
                except Exception as e:
                    # Log but continue for other tools
                    logger.debug(f"Could not load tool module {module_name}: {e}")

    def register(self, name: str, tool_class: Type[BaseAnalysisTool]):
        """Registers a tool class by its name."""
        self._tools[name] = tool_class

    def get_tool(self, name: str) -> Optional[BaseAnalysisTool]:
        """Returns an instance of a whitelisted tool. Auto-triggers discovery if needed."""
        self.discover_tools()
        tool_class = self._tools.get(name)
        if tool_class:
            return tool_class()
        return None

    def get_all_tool_metadata(self) -> List[Dict[str, Any]]:
        """Exports tool metadata for the Planner. Auto-triggers discovery."""
        self.discover_tools()
        metadata = []
        for name, tool_class in self._tools.items():
            metadata.append({
                "name": name,
                "description": getattr(tool_class, 'description', "No description available."),
                "is_hardened": getattr(tool_class, 'input_schema', None) is not None
            })
        return metadata

# Global accessor
tool_registry = ToolRegistry()
