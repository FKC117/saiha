from django.contrib import admin
from saiha.models import (
    Dataset, DatasetColumn, ToolCategory, Tool, 
    AnalysisSession, ChatMessage, SiteSettings, 
    AnalysisResult, AIAuditLog
)

@admin.register(SiteSettings)
class SiteSettingsAdmin(admin.ModelAdmin):
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

@admin.register(AnalysisResult)
class AnalysisResultAdmin(admin.ModelAdmin):
    list_display = ('tool_used', 'session', 'status', 'created_at')
    list_filter = ('status', 'tool_used')
    search_fields = ('query', 'error_message')

@admin.register(AIAuditLog)
class AIAuditLogAdmin(admin.ModelAdmin):
    list_display = ('model_id', 'session', 'tokens_input', 'tokens_output', 'timestamp')
    list_filter = ('model_id',)
