import os
import django
import sys

# Setup Django
sys.path.append(r'f:\saiha\alaina')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'alaina.settings')
django.setup()

from django.contrib.auth.models import User
from saiha.models import Corporate, CorporateProfile, UserQuota
from saiha.corporate_service import CorporateService

def test_corporate_flow():
    # 1. Cleanup
    User.objects.filter(email='corp_admin@test.com').delete()
    User.objects.filter(email='corp_user@test.com').delete()
    Corporate.objects.filter(name='Test Corp').delete()

    # 2. Setup Corporate
    admin = User.objects.create_user(username='corp_admin', email='corp_admin@test.com', password='password')
    corp = Corporate.objects.create(name='Test Corp', total_credits=100.0, rem_credits=100.0, max_users=5)
    CorporateProfile.objects.create(user=admin, corporate=corp, role=CorporateProfile.Role.ADMIN)

    print(f"Initial Pool: {corp.rem_credits} Credits")

    # 3. Add Member (Default 5 credits)
    user = User.objects.create_user(username='corp_user', email='corp_user@test.com', password='password')
    CorporateService.add_user_directly(corp, user)
    
    corp.refresh_from_db()
    quota = UserQuota.objects.get(user=user)
    
    print(f"After User Add - Pool: {corp.rem_credits} Credits")
    print(f"User Allocation: {quota.max_tokens / 10000.0} Credits")

    # 4. Reallocate to 20 Credits
    CorporateService.reallocate_credits(corp, user, 20.0)
    
    corp.refresh_from_db()
    quota.refresh_from_db()
    
    print(f"After Realloc - Pool: {corp.rem_credits} Credits")
    print(f"User Allocation: {quota.max_tokens / 10000.0} Credits")

    if corp.rem_credits == 80.0 and (quota.max_tokens / 10000.0) == 20.0:
        print("TEST SUCCESSFUL")
    else:
        print("TEST FAILED")

if __name__ == "__main__":
    test_corporate_flow()
