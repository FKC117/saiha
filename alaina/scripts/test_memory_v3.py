import os
import sys
import django
import json
import uuid
import hashlib
from pathlib import Path
from datetime import timedelta

# Resolve project root for imports
sys.path.append(str(Path(__file__).parent.parent))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'alaina.settings')
django.setup()

from saiha.models import AnalysisSession, Dataset, DatasetColumn, ChatMessage, AIAuditLog
from saiha.agents.analysis_agent import get_analysis_agent
from saiha.agents.memory_manager import MemoryManager
from saiha.llm_management.gemini_service import gemini_service
from django.contrib.auth.models import User

def verify_v3_system():
    print("--- STARTING V3 SYSTEM VERIFICATION ---")
    
    # 1. Setup Mock User & Session
    user, _ = User.objects.get_or_create(username="test_user", email="test@example.com")
    dataset, _ = Dataset.objects.get_or_create(
        user=user, 
        name="Test Analytics Dataset", 
        defaults={"file_size": 1024, "rows_count": 100, "columns_count": 5}
    )
    
    # Add some mock columns
    DatasetColumn.objects.get_or_create(dataset=dataset, column_name="Age", data_type="numeric", defaults={"column_index": 0})
    DatasetColumn.objects.get_or_create(dataset=dataset, column_name="Income", data_type="numeric", defaults={"column_index": 1})
    DatasetColumn.objects.get_or_create(dataset=dataset, column_name="Region", data_type="categorical", defaults={"column_index": 2})
    
    session, _ = AnalysisSession.objects.get_or_create(user=user, dataset=dataset, is_active=True)
    # Clear memory for a fresh test run
    session.memory_summary = ""
    session.working_memory = {}
    session.llm_cache_id = None
    session.llm_cache_hash = None
    session.save()
    
    print(f"Using Test Session: {session.id}")
    
    # 2. Test Step 1: Initial Query (Should trigger Cache Creation)
    query = "Show me the correlation matrix for Age and Income."
    agent = get_analysis_agent(session)
    
    # Padding static context to force caching threshold (> 8000 chars)
    from saiha.agents.analysis_planner import analysis_planner
    padding = "\n" + ("STRICT ANALYSIS RULE: Focus on statistical significance for every column. " * 150)
    analysis_planner._SYSTEM_INSTRUCTION_STATIC += padding

    print(f"\n[Test 1] Dispatching Query: '{query}'")
    agent.process_query(query)
    
    # Simulating Celery Worker Result & Interpretation
    from saiha.models import AnalysisResult
    result = AnalysisResult.objects.filter(session=session).last()
    if result:
        result.status = AnalysisResult.Status.SUCCESS
        result.result_data = {"data": {"Age": [1,2,3], "Income": [4,5,6]}, "artifacts": []}
        result.save()
        
        from saiha.agents.interpretation_agent import interpretation_agent
        print("Manually triggering Interpretation...")
        interpretation_agent.interpret_result(str(result.id))

    # Check session metadata
    session.refresh_from_db()
    print(f"Cache ID Created: {session.llm_cache_id}")
    print(f"Cache Hash: {session.llm_cache_hash}")
    
    # 3. Test Step 2: Follow-up Query (Should REUSE Cache and use Working Memory)
    follow_up = "Now plot it."
    print(f"\n[Test 2] Follow-up Query: '{follow_up}'")
    agent.process_query(follow_up)
    
    session.refresh_from_db()
    print(f"Working Memory: {json.dumps(session.working_memory, indent=2)}")
    
    # 4. Test Step 3: Trigger Summary Update
    print("\n[Test 3] Simulating messages to trigger Memory Summary...")
    for i in range(10): # Exceed the 5-message threshold
        ChatMessage.objects.create(session=session, message_type='user', content=f"Message {i}")
        ChatMessage.objects.create(session=session, message_type='assistant', content=f"Response {i}")
    
    # Next query should trigger summary first
    query3 = "What have we done so far?"
    print(f"Dispatching Query 3: '{query3}'")
    agent.process_query(query3)
    
    session.refresh_from_db()
    print(f"Memory Summary Generated: \n{session.memory_summary[:200]}...")
    
    if session.memory_summary:
        print("PASS: Memory Summary generated correctly.")
    else:
        print("FAIL: Memory Summary missing.")

    print("\n--- VERIFICATION COMPLETE ---")

if __name__ == "__main__":
    verify_v3_system()
