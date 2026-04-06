
import os
import sys
import django
import logging
import glob
import importlib
import inspect

# Setup Django environment
sys.path.append('d:/quantly/quanta')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'quanta.settings')
django.setup()

from quantalytics.ai_agents.tools.base_tool import BaseAnalysisTool

# Path to log file
LOG_FILE = r'd:/quantly/quanta/logs/tool_errors.log'

def clear_log():
    with open(LOG_FILE, 'w') as f:
        f.write('')

def verify_tools():
    tools_dir = r'd:/quantly/quanta/quantalytics/ai_agents/tools'
    sys.path.append(tools_dir)
    
    tool_files = glob.glob(os.path.join(tools_dir, "*.py"))
    
    success_count = 0
    fail_count = 0
    failed_tools = []
    
    print(f"Scanning tools in {tools_dir}...")
    
    for file_path in tool_files:
        filename = os.path.basename(file_path)
        if filename in ['__init__.py', 'base_tool.py', 'tool_registry.py', 'tool_parameters.py', 'parameter_validator.py', 'plot_utils.py', 'session_summary.py', 'smart_recommendations.py', 'apply_logging_fix.py', 'verify_tool_logging.py']:
            continue
            
        module_name = f"quantalytics.ai_agents.tools.{filename[:-3]}"
        
        try:
            module = importlib.import_module(module_name)
            
            # Find tool class
            tool_class = None
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, BaseAnalysisTool) and obj is not BaseAnalysisTool:
                    tool_class = obj
                    break
            
            if not tool_class:
                # print(f"Skipping {filename}: No tool class found.")
                continue
                
            tool_name = tool_class().name
            # print(f"Testing {tool_name} ({filename})...")
            
            # Instantiate tool
            tool = tool_class()
            
            # Monkey patch load_dataset or validate_dataset_requirement to raise exception
            # Most tools call validate_dataset_requirement or load_dataset early
            def mock_raise(*args, **kwargs):
                raise RuntimeError(f"Simulated Crash for {tool_name}")
                
            tool.validate_dataset_requirement = mock_raise
            tool.load_dataset = mock_raise
            
            # Clear log before run (optional, but finding specific line is better)
            # Run execute
            try:
                result = tool.execute(query="test")
            except Exception as e:
                print(f"❌ {tool_name}: UNCAUGHT Exception: {e}")
                fail_count += 1
                failed_tools.append(tool_name)
                continue
                
            # Check result status
            if result.get('status') != 'error':
                 print(f"❌ {tool_name}: Did not return error status. Got: {result.get('status')}")
                 fail_count += 1
                 failed_tools.append(tool_name)
                 continue

            # Check log file
            found_log = False
            with open(LOG_FILE, 'r') as f:
                log_content = f.read()
                if f"Simulated Crash for {tool_name}" in log_content:
                    found_log = True
                    
            if found_log:
                print(f"✅ {tool_name}: Verified.")
                success_count += 1
            else:
                print(f"❌ {tool_name}: Log entry NOT found.")
                fail_count += 1
                failed_tools.append(tool_name)

        except Exception as e:
            print(f"❌ Error testing {filename}: {e}")
            fail_count += 1

    print("\n" + "="*30)
    print(f"Summary: {success_count} Passed, {fail_count} Failed")
    if failed_tools:
        print(f"Failed Tools: {', '.join(failed_tools)}")

if __name__ == "__main__":
    # Ensure log directory exists
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    # clear_log() # Optional: start fresh
    verify_tools()
