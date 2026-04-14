import os
import django
from django.core.mail import send_mail
from django.conf import settings

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'alaina.settings')
django.setup()

def test_email_config():
    print(f"--- EMAIL CONFIGURATION ---")
    print(f"Backend: {settings.EMAIL_BACKEND}")
    print(f"Host: {settings.EMAIL_HOST}")
    print(f"Port: {settings.EMAIL_PORT}")
    print(f"User: {settings.EMAIL_HOST_USER or 'NOT SET'}")
    print(f"From: {settings.DEFAULT_FROM_EMAIL or 'NOT SET'}")
    
    try:
        print("\nAttempting to send test email...")
        send_mail(
            'ChatFlow System Test',
            'This is a diagnostic test of the email system.',
            settings.DEFAULT_FROM_EMAIL,
            ['test@example.com'], # Generic test email
            fail_silently=False,
        )
        print("SUCCESS: Email sent (check your terminal if using console backend, or your inbox if using SMTP).")
    except Exception as e:
        print(f"FAILED: {str(e)}")

if __name__ == "__main__":
    test_email_config()
