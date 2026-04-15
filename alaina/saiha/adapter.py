from allauth.account.adapter import DefaultAccountAdapter
from django.urls import reverse

class CustomAccountAdapter(DefaultAccountAdapter):
    def get_login_redirect_url(self, request):
        """
        Smart redirection after login.
        Corporate Admins -> Corporate Dashboard
        Everyone else -> Standard Index
        """
        user = request.user
        
        # Check for Corporate Profile
        profile = getattr(user, 'corp_profile', None)
        if profile and profile.is_active and profile.role == 'ADMIN':
            return reverse("saiha:corporate_dashboard")
            
        return reverse("saiha:index")
