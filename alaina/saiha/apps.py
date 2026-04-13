from django.apps import AppConfig


class SaihaConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "saiha"

    def ready(self):
        import saiha.signals
