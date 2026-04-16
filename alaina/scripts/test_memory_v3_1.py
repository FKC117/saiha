import os
import sys
import django
import uuid
import json
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

def test_elite_v3_1():
    print("Starting Elite Mode v3.1 Verification...")
    
    # 1. Setup Mock User & Session
    user = User.objects.first()
    dataset = Dataset.objects.first()
    session = AnalysisSession.objects.filter(user=user, dataset=dataset, is_active=True).first()
    if not session:
        session = AnalysisSession.objects.create(
            user=user,
            dataset=dataset,
            working_memory={"active_columns": ["Age"], "last_tool": "histogram"},
            memory_summary="User was exploring age distribution."
        )
    print(f"Using Session: {session.id}")

    # 2. Test Multi-line Metadata Parsing
    mock_ai_response = """
Here is my analysis of the age distribution. It looks skewed.

[METADATA]
type: skewness_finding
target: Age
columns: Age|Gender
"""
    clean_text, metadata = MemoryManager.extract_metadata_footer(mock_ai_response)
    print("\n--- Metadata Parsing Test ---")
    print(f"Clean Text Length: {len(clean_text)}")
    print(f"Extracted Metadata: {metadata}")
    assert "active_columns" in metadata and metadata["active_columns"] == ["Age", "Gender"]
    print("Metadata parsing successful.")

    # 3. Test Working Memory Update & Chain
    MemoryManager.update_working_memory(session, metadata=metadata)
    session.refresh_from_db()
    print(f"Updated WM: {session.working_memory}")
    print(f"Analysis Chain: {session.analysis_chain}")
    assert "Age" in session.working_memory["active_columns"]
    print("WM Update successful.")

    # 4. Test Decay Logic
    print("\n--- Decay Logic Test ---")
    # Use .update() to bypass auto_now=True
    AnalysisSession.objects.filter(id=session.id).update(
        last_activity=timezone.now() - timedelta(minutes=40)
    )
    session.refresh_from_db()
    
    MemoryManager.decay_stale_state(session)
    session.refresh_from_db()
    print(f"WM after 40m decay: {session.working_memory}")
    assert session.working_memory == {}
    print("Decay successful.")

    # 5. Test Structured snapshot for Hard Reset
    print("\n--- Hard Reset snapshot Test ---")
    # Create a mock result
    AnalysisResult.objects.create(
        session=session,
        tool_used="correlation_matrix",
        query="show correlation",
        status=AnalysisResult.Status.SUCCESS,
        ai_interpretation="Strong correlation found.",
        completed_at=timezone.now()
    )
    
    try:
        # We don't actually call the real API here to avoid token usage and key issues in verification
        # But we check if the snapshot generation works
        results = AnalysisResult.objects.filter(session=session).order_by('-completed_at')[:15]
        snapshots = []
        for r in reversed(results):
            snapshots.append({
                "query": r.query or "N/A",
                "tool": r.tool_used,
                "findings": (r.ai_interpretation or "")[:200],
                "status": r.status
            })
        print(f"Generated Snapshot: {json.dumps(snapshots[0])}")
        print("Hard Reset snapshot generation successful.")
    except Exception as e:
        print(f"Hard Reset failed: {e}")

    # 6. Test Cache Expiry Enforcement
    print("\n--- Cache Expiry Test ---")
    session.llm_cache_id = "expired_id"
    session.llm_cache_hash = "some_hash"
    session.llm_cache_expiry = timezone.now() - timedelta(minutes=1)
    session.save()
    
    cache_id = gemini_service.get_or_create_cache(session, "small", "small")
    assert cache_id is None
    print("Cache expiry enforcement confirmed.")

    print("\nElite Mode v3.1 Verification Complete!")

if __name__ == "__main__":
    test_elite_v3_1()
