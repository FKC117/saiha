from django.contrib import admin
from .models import Dataset, DatasetColumn, ToolCategory, Tool, AnalysisSession, ChatMessage, SiteSettings

@admin.register(SiteSettings)
class SiteSettingsAdmin(admin.ModelAdmin):
    """
    Ensures that only one instance of SiteSettings can be created.
    """
    def has_add_permission(self, request):
        if SiteSettings.objects.exists():
            return False
        return True

@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ('name', 'user', 'file_type', 'rows_count', 'upload_date')
    search_fields = ('name', 'user__email')

@admin.register(DatasetColumn)
class DatasetColumnAdmin(admin.ModelAdmin):
    list_display = ('dataset', 'column_name', 'data_type')

@admin.register(AnalysisSession)
class AnalysisSessionAdmin(admin.ModelAdmin):
    list_display = ('user', 'dataset', 'created_at', 'is_active')

@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ('session', 'message_type', 'created_at')

@admin.register(ToolCategory)
class ToolCategoryAdmin(admin.ModelAdmin):
    list_display = ('name', 'order')

@admin.register(Tool)
class ToolAdmin(admin.ModelAdmin):
    list_display = ('name', 'category', 'tool_type')
