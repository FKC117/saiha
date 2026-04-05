from .models import SiteSettings

def site_branding(request):
    """
    Adds site_settings to the template context globally.
    """
    try:
        settings = SiteSettings.objects.first()
        if not settings:
            # Fallback to defaults if no record exists
            settings = SiteSettings(site_name='ChatFlow')
    except Exception:
        settings = None
        
    return {'site_settings': settings}
