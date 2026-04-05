
import os
import sys
import django
import logging

# Setup Django environment
sys.path.append(r"d:\quantly\quanta")
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'quanta.settings')
django.setup()

def test_logging():
    print("Testing logging configuration...")
    
    # 1. Get the specific logger we are trying to use
    logger_name = "saiha.ai_agents.tools"
    logger = logging.getLogger(logger_name)
    
    print(f"Logger Name: {logger.name}")
    print(f"Effective Level: {logger.getEffectiveLevel()}")
    print(f"Handlers: {logger.handlers}")
    print(f"Parent: {logger.parent.name if logger.parent else 'None'}")
    
    # 2. Try to log a message
    test_msg = "TEST_LOG_ENTRY: If you see this in tool_errors.log, logging is working."
    try:
        logger.error(test_msg)
        print("Successfully sent log message.")
    except Exception as e:
        print(f"Failed to log message: {e}")

if __name__ == "__main__":
    test_logging()
