import os
import django
import sys

# Setup Django
sys.path.append(r'f:\saiha\alaina')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'alaina.settings')
django.setup()

from saiha.models import AIAuditLog, UserQuota, AnalysisSession, User
from django.db.models import Sum

print("--- USER REPORT ---")
users = User.objects.all()
for u in users:
    quota, _ = UserQuota.objects.get_or_create(user=u)
    personal_logs = AIAuditLog.objects.filter(user=u).count()
    total_tokens = AIAuditLog.objects.filter(user=u).aggregate(total=Sum('tokens_input') + Sum('tokens_output'))['total'] or 0
    print(f"User: {u.email} (ID: {u.id})")
    print(f"  Quota Record Tokens: {quota.current_tokens_used}")
    print(f"  Logs with User: {personal_logs}")
    print(f"  Tokens in those logs: {total_tokens}")
    print("-" * 20)

print("\n--- ORPHANED LOGS ---")
orphans = AIAuditLog.objects.filter(user__isnull=True)
print(f"Total Orphaned Logs: {orphans.count()}")
if orphans.exists():
    session_ids = orphans.values_list('session_id', flat=True).distinct()
    print(f"  Spanning Sessions: {list(session_ids)[:5]}...")
    
    # Try to find owners of these sessions
    valid_sessions = AnalysisSession.objects.filter(id__in=session_ids)
    for s in valid_sessions:
        print(f"  Session {s.id} is owned by: {s.user.email if s.user else 'NONE'}")
