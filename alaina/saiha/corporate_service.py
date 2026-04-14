import logging
from django.db import transaction
from django.utils import timezone
from datetime import timedelta
import uuid
from saiha.models import Corporate, CorporateProfile, CorporateInvitation, UserQuota, User, AppConfiguration

logger = logging.getLogger(__name__)

class CorporateService:
    @staticmethod
    def invite_user(corporate, email, first_name=None, last_name=None, initial_credits=5.0):
        """
        Creates a pending invitation for a user.
        Defensively checks seat limits.
        """
        # 1. Check seat limit
        current_members = corporate.members.filter(role=CorporateProfile.Role.MEMBER).count()
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
            first_name=first_name,
            last_name=last_name,
            token=token,
            initial_credits=initial_credits,
            expires_at=expires_at,
            status=CorporateInvitation.Status.PENDING
        )
        
        logger.info(f"Created invitation for {email} to join {corporate.name}")
        return invitation

    @staticmethod
    @transaction.atomic
    def add_user_directly(corporate, user, role=CorporateProfile.Role.MEMBER, initial_credits=5.0, first_name=None, last_name=None):
        """
        Links an existing user directly to a corporation.
        Subtracts from corporate unallocated pool.
        """
        # 1. Validation
        if CorporateProfile.objects.filter(user=user).exists():
            raise ValueError("User is already a member of a corporation.")
        
        # Only count MEMBERS against the seat limit
        if corporate.members.filter(role=CorporateProfile.Role.MEMBER).count() >= corporate.max_users:
            raise ValueError(f"Corporate seat limit reached ({corporate.max_users}).")
            
        if corporate.rem_credits < initial_credits:
            raise ValueError(f"Insufficient corporate credits to allocate {initial_credits}.")

        # 2. Update User Name if missing
        if first_name and not user.first_name:
            user.first_name = first_name
        if last_name and not user.last_name:
            user.last_name = last_name
        user.save()

        # 3. Create Profile
        profile = CorporateProfile.objects.create(
            user=user,
            corporate=corporate,
            role=role
        )

        # 3. Setup Quota
        quota, _ = UserQuota.objects.get_or_create(user=user)
        
        rate = AppConfiguration.get_rate()
        
        # Convert credits to tokens
        token_allocation = int(initial_credits * rate)
        quota.max_tokens = token_allocation
        quota.save()

        # 5. Deduct from Corporate Pool
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

    @staticmethod
    def resend_invitation(request, invitation_id):
        """
        Retrieves a pending invitation and resends it, updating timestamps.
        """
        invitation = CorporateInvitation.objects.get(id=invitation_id, is_accepted=False)
        return CorporateService.send_invitation_email(request, invitation)

    @staticmethod
    def send_invitation_email(request, invitation):
        """
        Sends the branded HTML invitation email and updates tracking fields.
        """
        from django.core.mail import EmailMultiAlternatives
        from django.template.loader import render_to_string
        from django.utils.html import strip_tags
        from django.conf import settings
        from django.utils import timezone

        # 1. Prepare Content
        context = {
            'corporate_name': invitation.corporate.name,
            'first_name': invitation.first_name,
            'initial_credits': invitation.initial_credits,
            'join_url': request.build_absolute_uri(f"/corporate/join/{invitation.id}/")
        }

        subject = f"Invitation to join {invitation.corporate.name} on ChatFlow"
        html_content = render_to_string('emails/corporate_invitation.html', context)
        text_content = strip_tags(html_content)

        # 2. Send Email
        try:
            msg = EmailMultiAlternatives(
                subject,
                text_content,
                settings.DEFAULT_FROM_EMAIL,
                [invitation.email]
            )
            msg.attach_alternative(html_content, "text/html")
            msg.send()
            
            # 3. Update Tracking
            now = timezone.now()
            if not invitation.sent_at:
                invitation.sent_at = now
            invitation.last_sent_at = now
            invitation.status = CorporateInvitation.Status.SENT
            invitation.save()
            
            logger.info(f"Sent invitation email to {invitation.email}")
            return True
        except Exception as e:
            logger.error(f"Failed to send invitation email to {invitation.email}: {e}")
            return False

    @staticmethod
    @transaction.atomic
    def discontinue_member(corporate, user):
        """
        Removes a member from the corporation.
        Recovers unspent credits and records the departure time.
        """
        profile = CorporateProfile.objects.get(user=user, corporate=corporate, is_active=True)
        quota = UserQuota.objects.get(user=user)
        
        # 1. Calculate unmanaged/unspent credits
        # We look at the actual credits remaining on their quota
        rate = float(AppConfiguration.get_rate())
        
        # credits_used is lifetime. We need to know how much of the CURRENT allocation is unspent.
        # However, for simplicity, we treat the remaining quota capacity as the recoverable amount.
        rem_quota_tokens = quota.max_tokens - quota.current_tokens_used
        rem_credits = max(0, rem_quota_tokens / rate)
        
        # 2. Add back to corporate pool
        corporate.rem_credits += float(rem_credits)
        corporate.save()
        
        # 3. Soft-deactivate the profile
        profile.is_active = False
        profile.left_at = timezone.now()
        profile.save()
        
        # 4. Reset User Quota to a default "Individual" state (e.g. 0 managed credits)
        quota.max_tokens = 0 
        quota.save()
        
        logger.info(f"Discontinued user {user.email} from {corporate.name}. Recovered {rem_credits} credits.")
        return True

    @staticmethod
    @transaction.atomic
    def purchase_seats(corporate, count):
        """
        Increases the corporate's max_users limit by deducting from their unallocated pool.
        """
        if count <= 0:
            raise ValueError("Select at least 1 seat to purchase.")

        # 1. Fetch Price
        try:
            config = AppConfiguration.objects.get()
            price_per_seat = config.credit_cost_per_seat
        except AppConfiguration.DoesNotExist:
            price_per_seat = 10.0 # Fallback

        total_cost = price_per_seat * count
        
        # 2. Verify Credits
        if corporate.rem_credits < total_cost:
            raise ValueError(f"Insufficient credits. Total cost for {count} seats is {total_cost} Credits.")

        # 3. Deduct and Expand
        corporate.rem_credits -= total_cost
        corporate.total_credits -= total_cost # This was spent on seats, no longer available for AI usage
        corporate.max_users += count
        corporate.save()

        logger.info(f"Corporate {corporate.name} purchased {count} seats for {total_cost} credits.")
        return True

    @staticmethod
    def sync_corporate_credits(corporate):
        """
        Checks if the corporate pool has expired and moves credits to the resuable pool.
        """
        from django.utils import timezone
        if corporate.expiry_date and timezone.now() > corporate.expiry_date:
            if corporate.rem_credits > 0:
                corporate.expired_credits += corporate.rem_credits
                corporate.rem_credits = 0
                corporate.save()
                logger.info(f"Corporate {corporate.name} credits expired. Moved {corporate.expired_credits} to rescue pool.")
        return corporate

    @staticmethod
    @transaction.atomic
    def recharge_corporate(corporate, amount_credits, days_valid=30):
        """
        Main entry point for adding credits. Performs the 'Rescue & Rollover'.
        """
        from django.utils import timezone
        from datetime import timedelta
        
        # 1. Prepare rescue
        rescued = corporate.expired_credits
        total_to_add = amount_credits + rescued
        
        # 2. Update Pool
        corporate.rem_credits += total_to_add
        corporate.total_credits += amount_credits # Total ledger only increases by the new purchase
        corporate.expired_credits = 0
        
        # 3. Extend Expiry
        new_expiry = timezone.now() + timedelta(days=days_valid)
        corporate.expiry_date = new_expiry
        
        corporate.save()
        logger.info(f"Recharged {corporate.name}: Added {amount_credits}, Rescued {rescued}. New Total: {corporate.rem_credits}")
        return total_to_add
