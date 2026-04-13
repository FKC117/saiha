from django import forms
from allauth.account.forms import SignupForm
from django.contrib.auth.models import User

class CustomSignupForm(SignupForm):
    first_name = forms.CharField(
        max_length=30, 
        label='First Name',
        widget=forms.TextInput(attrs={
            'placeholder': 'Enter first name',
            'required': True,
        })
    )
    last_name = forms.CharField(
        max_length=30, 
        label='Last Name',
        widget=forms.TextInput(attrs={
            'placeholder': 'Enter last name',
            'required': True,
        })
    )

    def save(self, request):
        user = super(CustomSignupForm, self).save(request)
        user.first_name = self.cleaned_data['first_name']
        user.last_name = self.cleaned_data['last_name']
        user.save()
        return user

