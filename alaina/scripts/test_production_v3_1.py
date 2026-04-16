import os
import sys
import django
import uuid
import json
import inspect
from datetime import timedelta
from django.utils import timezone

# Setup Django environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "alaina.settings")
django.setup()

from saiha.models import AnalysisSession, Dataset, AIAuditLog, AnalysisResult
from saiha.agents.memory_manager import MemoryManager
from saiha.llm_management.gemini_service import gemini_service
from django.contrib.auth.models import User

def test_production_v3_1():
    print("Starting Elite Mode v3.1 PRODUCTION Hardening Verification...")
    
    # 1. Setup Mock User & Session
    user = User.objects.first()
    dataset = Dataset.objects.first()
    session = AnalysisSession.objects.filter(user=user, dataset=dataset, is_active=True).first()
    if not session:
        session = AnalysisSession.objects.create(
            user=user,
            dataset=dataset,
            working_memory={"active_columns": ["Age"], "last_tool": "histogram", "last_result_type": "visualization"}
        )
    print(f"Using Session: {session.id}")

    # 2. Test Metadata Empty-Value Guard
    print("\n--- Metadata Guard Test ---")
    corrupt_ai_response = """
Here is my analysis.

[METADATA]
type: 
target: Income
columns: 
"""
    clean_text, metadata = MemoryManager.extract_metadata_footer(corrupt_ai_response)
    print(f"Parsed Metadata (should have skipped blanks): {metadata}")
    assert "active_columns" not in metadata
    assert "last_result_type" not in metadata
    assert metadata.get("last_target_column") == "Income"
    print("Metadata corruption guard successful.")

    # 3. Test Gradual Decay Logic
    print("\n--- Gradual Decay Test ---")
    session.working_memory = {"active_columns": ["Age"], "last_tool": "histogram"}
    session.save()
    
    # Use .update() to bypass auto_now=True
    AnalysisSession.objects.filter(id=session.id).update(
        last_activity=timezone.now() - timedelta(minutes=40)
    )
    session.refresh_from_db()
    
    MemoryManager.decay_stale_state(session)
    session.refresh_from_db()
    print(f"WM after 40m decay (should keep keys but values None): {session.working_memory}")
    assert session.working_memory.get("active_columns") is None
    print("Gradual decay successful.")

    # 4. Test Summary Quality Guard (Manual check of logic)
    print("\n--- Summary Guard Test ---")
    # Mocking llm_service to return bad summary
    class MockLLM:
        def generate_content(self, p): return "Too short"
    
    success = MemoryManager.update_summary(session, MockLLM())
    assert success is False
    print("Summary quality guard successful (rejected bad summary).")

    # 5. Test Cache Hash Versioning
    print("\n--- Cache Hash Versioning Test ---")
    source = inspect.getsource(gemini_service.get_or_create_cache)
    assert "TOOL_DESCRIPTIONS_VERSION" in source
    print("Cache Hash Versioning confirmed in source code.")

    print("\nElite Mode v3.1 PRODUCTION Hardening Verification Complete!")

if __name__ == "__main__":
    test_production_v3_1()
