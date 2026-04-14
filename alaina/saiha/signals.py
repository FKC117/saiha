from django.dispatch import receiver
from allauth.account.signals import user_signed_up
from allauth.socialaccount.models import SocialAccount

from django.db.models.signals import post_save
from django.contrib.auth.models import User
from .models import UserQuota

@receiver(user_signed_up)
def handle_user_signed_up(request, user, **kwargs):
    # Check if user signed up via social account (Google)
    is_social = SocialAccount.objects.filter(user=user).exists()
    
    if is_social:
        # Google users are pre-verified, so we can ensure their EmailAddress is marked as verified
        from allauth.account.models import EmailAddress
        email_address = EmailAddress.objects.filter(user=user, email=user.email).first()
        if email_address:
            email_address.verified = True
            email_address.save()

    # --- PENDING INVITATION LINKING ---
    pending_token = request.session.get('pending_invite_token')
    if pending_token:
        try:
            from .models import CorporateInvitation
            from .corporate_service import CorporateService
            invitation = CorporateInvitation.objects.filter(id=pending_token, is_accepted=False).first()
            if invitation:
                # Add to corp with stored name preferences
                CorporateService.add_user_directly(
                    invitation.corporate,
                    user,
                    first_name=invitation.first_name,
                    last_name=invitation.last_name,
                    initial_credits=invitation.initial_credits
                )
                invitation.is_accepted = True
                invitation.save()
                # Clear session
                del request.session['pending_invite_token']
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Failed to link invited user {user.email}: {e}")

@receiver(post_save, sender=User)
def ensure_user_quota(sender, instance, created, **kwargs):
    """
    Safety net: Ensure every user has a UserQuota record.
    """
    if created or not hasattr(instance, 'quota'):
        UserQuota.objects.get_or_create(user=instance)
