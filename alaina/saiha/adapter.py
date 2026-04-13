from allauth.account.adapter import DefaultAccountAdapter
from django.urls import reverse

class CustomAccountAdapter(DefaultAccountAdapter):
    def get_login_redirect_url(self, request):
        # We can still keep the safety check if we want, 
        # but allauth 'mandatory' verification usually does this automatically.
        return super().get_login_redirect_url(request)
