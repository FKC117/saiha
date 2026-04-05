import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'alaina.settings')
django.setup()

from django.contrib.sites.models import Site
from allauth.socialaccount.models import SocialApp
from django.contrib.auth.models import User

# 1. Update the default site
site, created = Site.objects.get_or_create(pk=1)
site.domain = '127.0.0.1:8000'
site.name = 'ChatFlow Local'
site.save()
print(f"{'Created' if created else 'Updated'} default site to 127.0.0.1:8000")

# 2. Setup Google SocialApp if credentials exist
client_id = os.getenv("GOOGLE_CLIENT_ID")
secret = os.getenv("GOOGLE_CLIENT_SECRET")

if client_id and secret and client_id != "your-client-id":
    app, created = SocialApp.objects.get_or_create(
        provider='google',
        name='Google Auth'
    )
    app.client_id = client_id
    app.secret = secret
    app.sites.add(site)
    app.save()
    print(f"{'Created' if created else 'Updated'} SocialApp for Google")
else:
    print("Skipping SocialApp setup: GOOGLE_CLIENT_ID/SECRET not properly configured in .env")

# 3. Create superuser if none exists
if not User.objects.filter(is_superuser=True).exists():
    User.objects.create_superuser('admin', 'admin@example.com', 'admin123')
    print("Created default superuser: admin / admin123")
