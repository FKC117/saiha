
import os
import glob
import re

TOOLS_DIR = r"d:/quantly/quanta/quantalytics/ai_agents/tools"

def verify_static():
    files = glob.glob(os.path.join(TOOLS_DIR, "*.py"))
    
    passed = []
    failed = []
    skipped = []
    
    print(f"Scanning {len(files)} files in {TOOLS_DIR}...\n")
    
    for file_path in files:
        filename = os.path.basename(file_path)
        
        # Skip init and helpers
        if filename.startswith('_') or filename in ['base_tool.py', 'tool_registry.py', 'tool_parameters.py', 'parameter_validator.py', 'plot_utils.py', 'session_summary.py', 'smart_recommendations.py', 'apply_logging_fix.py', 'verify_tool_logging.py', 'static_verify_logging.py']:
            skipped.append(filename)
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if "class " not in content or "BaseAnalysisTool" not in content:
            skipped.append(filename)
            continue
            
        # Check for try-except block in execute/run
        has_try = "try:" in content
        has_except = "except Exception as " in content
        has_log = "self.log_error(" in content
        
        if has_try and has_except:
            if has_log:
                passed.append(filename)
            else:
                failed.append(filename)
        else:
            # Some tools might not have a try-except block at all (rare/bad)
            failed.append(f"{filename} (No try/except block found)")

    print("="*40)
    print(f"VERIFICATION SUMMARY")
    print("="*40)
    print(f"Total Tools Checked: {len(passed) + len(failed)}")
    print(f"Passed: {len(passed)}")
    print(f"Failed: {len(failed)}")
    print("-" * 20)
    
    if failed:
        print("\nFAILED TOOLS (Action Required):")
        for f in failed:
            print(f"❌ {f}")
            
    if passed:
        # print("\nPASSED TOOLS:")
        # for f in passed:
        #     print(f"✅ {f}")
        pass

if __name__ == "__main__":
    verify_static()
