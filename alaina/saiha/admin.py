from django.contrib import admin
from saiha.models import (
    Dataset, DatasetColumn, ToolCategory, Tool, 
    AnalysisSession, ChatMessage, SiteSettings, 
    AnalysisResult, AIAuditLog, UserQuota, AppConfiguration,
    Corporate, CorporateProfile, CorporateInvitation, CreditPackage,
    Invoice, BusinessInfo
)

@admin.register(CreditPackage)
class CreditPackageAdmin(admin.ModelAdmin):
    list_display = ('name', 'credits', 'price_usd', 'price_bdt', 'is_popular', 'is_active')
    list_editable = ('price_usd', 'price_bdt', 'is_popular', 'is_active')
    search_fields = ('name',)
    list_per_page = 10
    

@admin.register(Corporate)
class CorporateAdmin(admin.ModelAdmin):
    list_display = ('name', 'total_credits', 'rem_credits', 'max_users', 'created_at')
    search_fields = ('name',)
    list_per_page = 10

@admin.register(CorporateProfile)
class CorporateProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'corporate', 'role', 'is_active')
    list_filter = ('role', 'corporate')
    search_fields = ('user__email', 'corporate__name')
    list_per_page = 10

@admin.register(CorporateInvitation)
class CorporateInvitationAdmin(admin.ModelAdmin):
    list_display = ('email', 'corporate', 'is_accepted', 'expires_at')
    list_filter = ('is_accepted', 'corporate')
    list_per_page = 10

@admin.register(AppConfiguration)
class AppConfigurationAdmin(admin.ModelAdmin):
    list_display = ('id', 'token_to_credit_rate', 'credit_cost_per_seat', 'default_vat_percentage', 'updated_at')
    
    def has_add_permission(self, request):
        if AppConfiguration.objects.exists():
            return False
        return True

@admin.register(UserQuota)
class UserQuotaAdmin(admin.ModelAdmin):
    list_display = ('user', 'plan_name', 'current_tokens_used', 'max_tokens', 'expiry_date')
    search_fields = ('user__email', 'plan_name')
    list_per_page = 10

@admin.register(SiteSettings)
class SiteSettingsAdmin(admin.ModelAdmin):
    def has_add_permission(self, request):
        if SiteSettings.objects.exists():
            return False
        return True

@admin.register(BusinessInfo)
class BusinessInfoAdmin(admin.ModelAdmin):
    list_display = ('company_name', 'email', 'phone', 'bin_vat_number')
    def has_add_permission(self, request):
        if BusinessInfo.objects.exists():
            return False
        return True

@admin.register(Invoice)
class InvoiceAdmin(admin.ModelAdmin):
    list_display = ('invoice_number', 'get_target', 'description', 'total_amount_local', 'currency', 'status', 'created_at')
    list_filter = ('status', 'currency', 'created_at')
    search_fields = ('invoice_number', 'user__email', 'corporate__name', 'description')
    readonly_fields = ('invoice_number', 'created_at')
    list_per_page = 10

    def get_target(self, obj):
        return obj.corporate.name if obj.corporate else obj.user.email
    get_target.short_description = 'Customer'

@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ('name', 'user', 'file_type', 'rows_count', 'upload_date')
    search_fields = ('name', 'user__email')
    list_per_page = 10

@admin.register(DatasetColumn)
class DatasetColumnAdmin(admin.ModelAdmin):
    list_display = ('dataset', 'column_name', 'data_type')
    list_per_page = 10

@admin.register(AnalysisSession)
class AnalysisSessionAdmin(admin.ModelAdmin):
    list_display = ('user', 'dataset', 'llm_cache_id', 'created_at', 'is_active')
    list_filter = ('is_active', 'created_at')
    search_fields = ('user__email', 'llm_cache_id')
    readonly_fields = ('memory_summary', 'working_memory', 'analysis_chain', 'llm_cache_id', 'llm_cache_hash', 'llm_cache_expiry')
    list_per_page = 10

@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ('session', 'message_type', 'created_at')
    list_per_page = 10

@admin.register(ToolCategory)
class ToolCategoryAdmin(admin.ModelAdmin):
    list_display = ('name', 'order')
    list_per_page = 10

@admin.register(Tool)
class ToolAdmin(admin.ModelAdmin):
    list_display = ('name', 'category', 'tool_type')
    list_per_page = 10

@admin.register(AnalysisResult)
class AnalysisResultAdmin(admin.ModelAdmin):
    list_display = ('tool_used', 'session', 'status', 'created_at')
    list_filter = ('status', 'tool_used')
    search_fields = ('query', 'error_message')
    list_per_page = 10

@admin.register(AIAuditLog)
class AIAuditLogAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'session', 'model_id', 'tokens_cached', 'cache_hit', 'timestamp')
    list_filter = ('model_id', 'cache_hit', 'timestamp')
    readonly_fields = ('prompt', 'response', 'working_memory_snapshot', 'timestamp')
    list_per_page = 10
