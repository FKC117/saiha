from django.db import migrations, models
from django.db.models import Q


class Migration(migrations.Migration):

    dependencies = [
        ('saiha', '0022_creditrequest'),
    ]

    operations = [
        migrations.AlterUniqueTogether(
            name='analysissession',
            unique_together=set(),
        ),
        migrations.AddConstraint(
            model_name='analysissession',
            constraint=models.UniqueConstraint(
                fields=('user', 'dataset'),
                condition=Q(is_active=True),
                name='unique_active_session_per_user_dataset',
            ),
        ),
    ]
