import os
import django
import sys
import json

# Setup Django
sys.path.append(r'f:\saiha\alaina')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'alaina.settings')
django.setup()

from saiha.views import get_usage_data
from django.test import RequestFactory
from saiha.models import User

user = User.objects.get(email='fazlul.karim117@gmail.com')
factory = RequestFactory()
request = factory.get('/usage/stats/')
request.user = user

response = get_usage_data(request)
print(json.dumps(json.loads(response.content), indent=2))
