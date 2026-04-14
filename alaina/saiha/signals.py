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

@receiver(post_save, sender=User)
def ensure_user_quota(sender, instance, created, **kwargs):
    """
    Safety net: Ensure every user has a UserQuota record.
    """
    if created or not hasattr(instance, 'quota'):
        UserQuota.objects.get_or_create(user=instance)
