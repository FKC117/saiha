import os
import sys
import django

# Add current directory to path
sys.path.append(os.getcwd())
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'alaina.settings')
django.setup()

from saiha.models import CreditPackage

packages = [
    {'name': 'Starter', 'credits': 50, 'price_usd': 5.00, 'price_bdt': 600, 'is_popular': False},
    {'name': 'Professional', 'credits': 250, 'price_usd': 25.00, 'price_bdt': 3000, 'is_popular': True},
    {'name': 'Enterprise', 'credits': 1000, 'price_usd': 100.00, 'price_bdt': 11500, 'is_popular': False},
]

for p in packages:
    obj, created = CreditPackage.objects.update_or_create(
        name=p['name'],
        defaults={
            'credits': p['credits'], 
            'price_usd': p['price_usd'], 
            'price_bdt': p['price_bdt'],
            'is_popular': p['is_popular']
        }
    )
    print(f"Updated/Created package: {obj.name} with BDT price: {obj.price_bdt}")
