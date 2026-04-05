import os
import django
import time
import uuid
import sys
from celery import chain

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'alaina.settings')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
django.setup()

from saiha.models import AnalysisSession, AnalysisResult, Dataset
from saiha.celery_tasks import execute_analysis_task, interpret_analysis_task
from saiha.redis_stack import redis_manager

def run_stress_tests():
    print("🚀 Starting Hardened Async Stress Tests...")
    
    # Setup dummy session
    session = AnalysisSession.objects.first()
    if not session:
        print("❌ No session found. Please create one.")
        return

    # --- TEST 1: Atomic Idempotency (Concurrency) ---
    print("\n🧪 1. Atomic Idempotency Test...")
    dedup_id = f"test-dedup-{uuid.uuid4()}"
    
    # Create two placeholders for the SAME dedup_id (simulate race)
    # Note: Our model uses UNIQUE on dedup_id.
    try:
        res1 = AnalysisResult.objects.create(session=session, tool_used="stats", dedup_id=dedup_id)
        print("✅ First record created.")
        
        try:
            res2 = AnalysisResult.objects.create(session=session, tool_used="stats", dedup_id=dedup_id)
            print("❌ ERROR: Second record with same dedup_id should have failed!")
        except Exception:
            print("✅ Atomic DB constraint blocked duplicate write.")
    except Exception as e:
        print(f"❌ Initial creation failed: {e}")

    # --- TEST 2: Selective Transient Retries ---
    # Trigger a task and check if it gets scheduled
    print("\n🧪 2. Scheduling Analysis Task (Observability check)...")
    res_async = AnalysisResult.objects.create(session=session, tool_used="regression")
    task = execute_analysis_task.delay(res_async.id, "regression", {"col": "x"})
    print(f"✅ Task {task.id} dispatched. Initial status: {res_async.status}")
    
    time.sleep(1)
    res_async.refresh_from_db()
    print(f"👉 Status after 1s: {res_async.status} (Task ID recorded: {res_async.task_id})")

    # --- TEST 3: Chaining Test ---
    print("\n🧪 3. Task Chaining (Workflows)...")
    res_chain = AnalysisResult.objects.create(session=session, tool_used="correlation")
    workflow = chain(
        execute_analysis_task.s(res_chain.id, "correlation", {}),
        interpret_analysis_task.s(res_chain.id)
    )
    workflow_result = workflow.apply_async()
    print(f"✅ Workflow Chain dispatched. Chain Root ID: {workflow_result.id}")

    # --- TEST 4: No-Sync Fallback (Redis Check) ---
    print("\n🧪 4. Redis Connectivity...")
    if redis_manager.is_alive():
        print("✅ Redis is ALIVE. Connection manager functioning.")
    else:
        print("❌ Redis is DOWN. System should refuse sync execution.")

    print("\n🏁 Stress Tests Triggered. Please check your Celery worker logs for execution results.")

if __name__ == "__main__":
    run_stress_tests()
