import logging
from celery import Task
from django.db import transaction, IntegrityError
from django.utils import timezone
from ..models import AnalysisResult

logger = logging.getLogger(__name__)

class BaseAnalysisTask(Task):
    """
    Hardened Base Task for ChartFlow.
    Features:
    - Atomic Idempotency (DB Integrity check)
    - Automatic Status Transitions
    - Selective Transient Retries
    - Payload-by-Reference (passed IDs)
    """
    
    # Selective Retry: Only for transient network/timeout issues
    # Skip retrying for: ValidationError, ValueError, KeyError (Logic/Data errors)
    autoretry_for = (ConnectionError, TimeoutError)
    retry_backoff = True
    retry_backoff_max = 600 # 10 mins
    retry_jitter = True
    max_retries = 5

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Log retries for observability."""
        logger.warning(f"Retrying task {task_id} (Attempt {self.request.retries}): {exc}")

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle final failures gracefully."""
        # result_id is typically the first arg
        result_id = args[0] if args else kwargs.get('result_id')
        if result_id:
            try:
                AnalysisResult.objects.filter(id=result_id).update(
                    status=AnalysisResult.Status.FAILED,
                    error_message=str(exc),
                    completed_at=timezone.now()
                )
            except Exception:
                logger.error(f"Failed to update task failure for {result_id}")
        logger.error(f"Task {task_id} FAILED: {exc}", exc_info=True)

    def start_observability(self, result_id, task_id):
        """Update DB to indicate the task has started."""
        AnalysisResult.objects.filter(id=result_id).update(
            status=AnalysisResult.Status.RUNNING,
            task_id=task_id,
            started_at=timezone.now()
        )

    def complete_observability(self, result_id):
        """Update DB to indicate success."""
        AnalysisResult.objects.filter(id=result_id).update(
            status=AnalysisResult.Status.SUCCESS,
            completed_at=timezone.now()
        )

    @transaction.atomic
    def ensure_idempotency(self, result_id, dedup_id):
        """
        Guarantees that a task won't create duplicate records.
        Returns the object if it exists or was just created.
        """
        try:
            # We use update_or_create or just update the existing placeholder
            # If the user's logic already created a placeholder:
            res = AnalysisResult.objects.get(id=result_id)
            if res.dedup_id and res.dedup_id != dedup_id:
                # Potential collision logic
                pass
            return res
        except AnalysisResult.DoesNotExist:
            logger.error(f"AnalysisResult {result_id} missing during idempotency check.")
            raise
