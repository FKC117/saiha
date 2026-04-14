import logging
from django.db import transaction
from django.utils import timezone
from datetime import timedelta
import uuid
from .models import Corporate, CorporateProfile, CorporateInvitation, UserQuota, User

logger = logging.getLogger(__name__)

class CorporateService:
    @staticmethod
    def invite_user(corporate, email, initial_credits=5.0):
        """
        Creates a pending invitation for a user.
        Defensively checks seat limits.
        """
        # 1. Check seat limit
        current_members = corporate.members.count()
        pending_invites = corporate.invitations.filter(is_accepted=False, expires_at__gt=timezone.now()).count()
        
        if current_members + pending_invites >= corporate.max_users:
            raise ValueError(f"Corporate seat limit reached ({corporate.max_users}).")

        # 2. Check credit pool
        if corporate.rem_credits < initial_credits:
            raise ValueError(f"Insufficient corporate credits to allocate {initial_credits} as default.")

        # 3. Create Invitation
        token = str(uuid.uuid4())
        expires_at = timezone.now() + timedelta(days=7)
        
        invitation = CorporateInvitation.objects.create(
            corporate=corporate,
            email=email,
            token=token,
            initial_credits=initial_credits,
            expires_at=expires_at
        )
        
        logger.info(f"Created invitation for {email} to join {corporate.name}")
        return invitation

    @staticmethod
    @transaction.atomic
    def add_user_directly(corporate, user, role=CorporateProfile.Role.MEMBER, initial_credits=5.0):
        """
        Links an existing user directly to a corporation.
        Subtracts from corporate unallocated pool.
        """
        # 1. Validation
        if CorporateProfile.objects.filter(user=user).exists():
            raise ValueError("User is already a member of a corporation.")
        
        if corporate.members.count() >= corporate.max_users:
            raise ValueError(f"Corporate seat limit reached ({corporate.max_users}).")
            
        if corporate.rem_credits < initial_credits:
            raise ValueError(f"Insufficient corporate credits to allocate {initial_credits}.")

        # 2. Create Profile
        profile = CorporateProfile.objects.create(
            user=user,
            corporate=corporate,
            role=role
        )

        # 3. Setup Quota
        quota, _ = UserQuota.objects.get_or_create(user=user)
        
        from .models import AppConfiguration
        rate = AppConfiguration.get_rate()
        
        # Convert credits to tokens
        token_allocation = int(initial_credits * rate)
        quota.max_tokens = token_allocation
        quota.save()

        # 4. Deduct from Corporate Pool
        corporate.rem_credits -= float(initial_credits)
        corporate.save()

        logger.info(f"User {user.email} added to {corporate.name} with {initial_credits} credits.")
        return profile

    @staticmethod
    @transaction.atomic
    def reallocate_credits(corporate, target_user, new_credit_limit):
        """
        Allows Corporate Admin to change a user's credit limit.
        Adjusts Corporate pool accordingly.
        """
        profile = CorporateProfile.objects.get(user=target_user, corporate=corporate)
        quota = UserQuota.objects.get(user=target_user)
        
        from .models import AppConfiguration
        rate = AppConfiguration.get_rate()
        
        current_credits = quota.max_tokens / float(rate)
        diff = new_credit_limit - current_credits

        if diff > corporate.rem_credits:
            raise ValueError(f"Insufficient corporate pool. Needed {diff}, available {corporate.rem_credits}")

        # Apply changes
        quota.max_tokens = int(new_credit_limit * rate)
        quota.save()

        corporate.rem_credits -= float(diff)
        corporate.save()

        logger.info(f"Reallocated credits for {target_user.email}: {current_credits} -> {new_credit_limit}")
        return True
