from django.dispatch import receiver
from allauth.account.signals import user_signed_up
from allauth.socialaccount.models import SocialAccount

@receiver(user_signed_up)
def handle_user_signed_up(request, user, **kwargs):
    # Check if user signed up via social account (Google)
    is_social = SocialAccount.objects.filter(user=user).exists()
    
    if not is_social:
        # standard allauth verification link will be sent automatically
        pass
    else:
        # Google users are pre-verified, so we can ensure their EmailAddress is marked as verified
        from allauth.account.models import EmailAddress
        email_address = EmailAddress.objects.filter(user=user, email=user.email).first()
        if email_address:
            email_address.verified = True
            email_address.save()
