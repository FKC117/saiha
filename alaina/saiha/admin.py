from django.contrib import admin
from saiha.models import (
    Dataset, DatasetColumn, ToolCategory, Tool, 
    AnalysisSession, ChatMessage, SiteSettings, 
    AnalysisResult, AIAuditLog, UserQuota, AppConfiguration,
    Corporate, CorporateProfile, CorporateInvitation, CreditPackage
)

@admin.register(CreditPackage)
class CreditPackageAdmin(admin.ModelAdmin):
    list_display = ('name', 'credits', 'price_usd', 'price_bdt', 'is_popular', 'is_active')
    list_editable = ('price_usd', 'price_bdt', 'is_popular', 'is_active')
    search_fields = ('name',)

@admin.register(Corporate)
class CorporateAdmin(admin.ModelAdmin):
    list_display = ('name', 'total_credits', 'rem_credits', 'max_users', 'created_at')
    search_fields = ('name',)

@admin.register(CorporateProfile)
class CorporateProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'corporate', 'role', 'is_active')
    list_filter = ('role', 'corporate')
    search_fields = ('user__email', 'corporate__name')

@admin.register(CorporateInvitation)
class CorporateInvitationAdmin(admin.ModelAdmin):
    list_display = ('email', 'corporate', 'is_accepted', 'expires_at')
    list_filter = ('is_accepted', 'corporate')

@admin.register(AppConfiguration)
class AppConfigurationAdmin(admin.ModelAdmin):
    list_display = ('id', 'token_to_credit_rate', 'updated_at')
    
    def has_add_permission(self, request):
        if AppConfiguration.objects.exists():
            return False
        return True

@admin.register(UserQuota)
class UserQuotaAdmin(admin.ModelAdmin):
    list_display = ('user', 'plan_name', 'current_tokens_used', 'max_tokens', 'expiry_date')
    search_fields = ('user__email', 'plan_name')

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
