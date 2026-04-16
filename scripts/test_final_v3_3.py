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

from saiha.models import AnalysisSession, Dataset, User, DatasetColumn
from saiha.agents.context_resolver import ContextResolver
from saiha.agents.memory_manager import MemoryManager
from saiha.agents.analysis_planner import analysis_planner
from saiha.agents.context_builder import ContextBuilder

def run_verification():
    print("--- Elite Mode v3.3 Final Polish Verification ---")
    
    # 1. Setup Mock User & Dataset
    user, _ = User.objects.get_or_create(username="test_pro", email="pro@test.com")
    dataset, _ = Dataset.objects.get_or_create(
        user=user, 
        name="Production Data",
        defaults={"file_size": 1024, "rows_count": 100, "columns_count": 5}
    )
    DatasetColumn.objects.get_or_create(dataset=dataset, column_name="Annual_Income", defaults={"column_index": 0})
    DatasetColumn.objects.get_or_create(dataset=dataset, column_name="Patient_Age", defaults={"column_index": 1})
    
    session, _ = AnalysisSession.objects.get_or_create(user=user, dataset=dataset)
    MemoryManager.reset_memory(session)

    # 2. Test Normalization & Fuzzy Signal Detection (Pivoting)
    session.working_memory = {"active_columns": ["Patient_Age"]}
    session.save()
    
    # Query mentions "income" (matches Annual_Income normalized)
    q_pivot = "show me income"
    has_signal = analysis_planner._query_has_new_signal(q_pivot, session)
    print(f"OK: Fuzzy Signal Detection (income -> Annual_Income): {has_signal} (Expected True)")
    assert has_signal == True

    is_follow_up = analysis_planner._is_follow_up(q_pivot, session)
    print(f"OK: Pivot blocking follow-up: {is_follow_up} (Expected False)")
    assert is_follow_up == False

    # 3. Test Context-Aware Override
    # False Positive check: "not sure"
    q_neg = "I am not sure what that means"
    is_ov_neg = analysis_planner._is_override(q_neg, session)
    print(f"OK: Override False Positive check ('not sure'): {is_ov_neg} (Expected False)")
    assert is_ov_neg == False

    # True Positive check: "no use income"
    q_pos = "no, use income instead"
    is_ov_pos = analysis_planner._is_override(q_pos, session)
    print(f"OK: Override True Positive check ('no use income'): {is_ov_pos} (Expected True)")
    assert is_ov_pos == True

    # 4. Test Deterministic History Cap (Elite v3.3)
    # Create 10 messages, some with metadata, some without
    from saiha.models import ChatMessage
    ChatMessage.objects.filter(session=session).delete()
    for i in range(10):
        m = ChatMessage.objects.create(
            session=session,
            message_type='ai' if i % 2 == 0 else 'user',
            content=f"Turn {i}",
            metadata={"id": "test"} if i % 2 == 0 else {}
        )
    
    history = ContextBuilder._get_recent_messages(session, limit=5)
    turn_count = len(history.split("\n"))
    print(f"OK: History Cap (5 turns): {turn_count} (Expected 5)")
    assert turn_count <= 5

    # 5. Test Metadata Completeness (Quality-Aware Priority)
    partial_meta = {"active_columns": ["Age"]} # Missing 'last_result_type'
    is_comp = ContextResolver.is_complete(partial_meta)
    print(f"OK: Metadata Completeness check (Partial): {is_comp} (Expected False)")
    assert is_comp == False

    complete_meta = {"active_columns": ["Age"], "last_result_type": "plot"}
    is_comp_full = ContextResolver.is_complete(complete_meta)
    print(f"OK: Metadata Completeness check (Complete): {is_comp_full} (Expected True)")
    assert is_comp_full == True

    print("\n--- ALL FINAL MICRO-GAPS VERIFIED ---")

if __name__ == "__main__":
    run_verification()
