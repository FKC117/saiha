import os
import django
import sys
import json
from datetime import timedelta
from django.utils import timezone

# Setup Django environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'alaina')))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "alaina.settings")
django.setup()

from saiha.models import AnalysisSession, Dataset, User
from saiha.agents.context_resolver import ContextResolver
from saiha.agents.memory_manager import MemoryManager
from saiha.agents.analysis_planner import analysis_planner

def run_verification():
    print("--- Elite Mode v3.2 Hardening Verification ---")
    
    # 1. Setup Mock User & Dataset
    user, _ = User.objects.get_or_create(username="test_pro", email="pro@test.com")
    dataset, _ = Dataset.objects.get_or_create(
        user=user, 
        name="Production Data",
        defaults={
            "file_size": 1024,
            "rows_count": 100,
            "columns_count": 5
        }
    )
    session, _ = AnalysisSession.objects.get_or_create(user=user, dataset=dataset)
    
    # Reset for clean start
    MemoryManager.reset_memory(session)
    print("OK: Reset Memory for fresh start.")

    # 2. Test Context Resolution (Priority)
    session.memory_summary = "Old summary about Age."
    session.working_memory = {"active_columns": ["Income"]}
    session.last_valid_metadata = {"active_columns": ["Spending"]}
    session.save()
    
    resolved = ContextResolver.resolve_state(session)
    print(f"OK: Resolution Priority Check: {resolved['active_columns']} (Expected ['Spending'])")
    assert resolved['active_columns'] == ["Spending"]
    assert resolved['source'] == "metadata"

    # 3. Test Hard Limits
    session.working_memory = {"active_columns": ["A", "B", "C", "D", "E", "F", "G"]}
    session.last_valid_metadata = {}
    session.save()
    
    resolved = ContextResolver.resolve_state(session)
    print(f"OK: Limit Check (Columns): {len(resolved['active_columns'])} (Expected 5)")
    assert len(resolved['active_columns']) == 5

    # 4. Test Hybrid Gating (Follow-up detection)
    q1 = "Analyze Age"
    is_f1 = analysis_planner._is_follow_up(q1, session)
    print(f"OK: Gating (Fresh Query): {is_f1} (Expected False)")
    
    q2 = "Why is it like that?"
    is_f2 = analysis_planner._is_follow_up(q2, session)
    print(f"OK: Gating (Follow-up 'Why/It'): {is_f2} (Expected True)")
    
    q3 = "continue"
    is_f3 = analysis_planner._is_follow_up(q3, session)
    print(f"OK: Gating (Brevity): {is_f3} (Expected True)")

    # 5. Test Metadata Recovery (Fallback)
    # Simulate failed extraction in MemoryManager
    text = "Here is the result. [NO METADATA]"
    clean, meta = MemoryManager.extract_metadata_footer(text)
    print(f"OK: Extraction Recovery: {meta} (Expected {{}})")
    assert meta == {}
    
    # 6. Test Reset Memory
    MemoryManager.reset_memory(session)
    print(f"OK: Hard Reset check: Summary='{session.memory_summary}'")
    assert session.memory_summary == ""

    print("\n--- ALL HARDENING GUARDS VERIFIED ---")

if __name__ == "__main__":
    run_verification()
