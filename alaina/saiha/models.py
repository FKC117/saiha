from django.db import models
from django.contrib.auth.models import User
import uuid
import json
from django.utils import timezone

class Dataset(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    original_filename = models.CharField(max_length=255)
    file_type = models.CharField(max_length=50)  # xlsx, csv, json
    storage_format = models.CharField(max_length=20, default='parquet')
    file_size = models.BigIntegerField()
    rows_count = models.IntegerField()
    columns_count = models.IntegerField()
    encoding = models.CharField(max_length=50, default='utf-8')
    upload_date = models.DateTimeField(auto_now_add=True)
    last_modified = models.DateTimeField(auto_now=True)
    is_processed = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    
    # File paths
    original_file_path = models.CharField(max_length=500, null=True, blank=True)
    processed_file_path = models.CharField(max_length=500)
    metadata_file_path = models.CharField(max_length=500)
    preview_file_path = models.CharField(max_length=500, null=True, blank=True)
    
    # Lineage Tracking
    parent_dataset = models.ForeignKey('self', on_delete=models.SET_NULL, null=True, blank=True, related_name='derived_datasets')
    
    class Meta:
        ordering = ['-upload_date']
        verbose_name = 'Dataset'
        verbose_name_plural = 'Datasets'
    
    def __str__(self):
        return f"{self.name} ({self.file_type})"
    
    def get_column_types(self):
        return {col.column_name: col.data_type for col in self.columns.all()}

    def get_numeric_columns(self):
        return [col.column_name for col in self.columns.filter(data_type__in=['integer', 'float'])]
    
    def get_categorical_columns(self):
        return [col.column_name for col in self.columns.filter(data_type__in=['string', 'boolean'])]
    
    def get_date_columns(self):
        return [col.column_name for col in self.columns.filter(data_type='date')]

class DatasetColumn(models.Model):
    DATA_TYPE_CHOICES = [
        ('string', 'String'),
        ('integer', 'Integer'),
        ('float', 'Float'),
        ('date', 'Date'),
        ('boolean', 'Boolean'),
    ]
    
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='columns')
    column_name = models.CharField(max_length=255)
    column_index = models.IntegerField()
    data_type = models.CharField(max_length=50, choices=DATA_TYPE_CHOICES, default='string')
    null_count = models.IntegerField(default=0)
    unique_count = models.IntegerField(default=0)
    sample_values = models.JSONField(default=list)
    
    class Meta:
        ordering = ['column_index']
        verbose_name = 'Dataset Column'
        verbose_name_plural = 'Dataset Columns'
    
    def __str__(self):
        return f"{self.dataset.name} - {self.column_name}"

class ToolCategory(models.Model):
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True, null=True)
    icon = models.CharField(max_length=50, default='fas fa-tools')
    color = models.CharField(max_length=20, default='#007bff')
    order = models.PositiveIntegerField(default=0)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['order', 'name']
        verbose_name = 'Tool Category'
        verbose_name_plural = 'Tool Categories'
    
    def __str__(self):
        return self.name

class Tool(models.Model):
    TOOL_TYPE_CHOICES = [
        ('statistical', 'Statistical Analysis'),
        ('visualization', 'Data Visualization'),
        ('data_quality', 'Data Quality'),
        ('machine_learning', 'Machine Learning'),
        ('export', 'Export/Import'),
        ('other', 'Other'),
    ]
    
    category = models.ForeignKey(ToolCategory, on_delete=models.CASCADE, related_name='tools')
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True, null=True)
    tool_type = models.CharField(max_length=50, choices=TOOL_TYPE_CHOICES, default='statistical')
    icon = models.CharField(max_length=50, default='fas fa-chart-bar')
    color = models.CharField(max_length=20, default='#28a745')
    order = models.PositiveIntegerField(default=0)
    is_active = models.BooleanField(default=True)
    requires_dataset = models.BooleanField(default=True)
    api_endpoint = models.CharField(max_length=200, blank=True, null=True)
    
    class Meta:
        ordering = ['category__order', 'order', 'name']
        verbose_name = 'Tool'
        verbose_name_plural = 'Tools'
    
    def __str__(self):
        return f"{self.category.name} - {self.name}"

class AnalysisSession(models.Model):
    """Analysis session linking user, dataset, and chat history."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, null=True, blank=True)
    session_name = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    last_activity = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    
    # Session metadata tracking
    analysis_count = models.PositiveIntegerField(default=0)
    chat_message_count = models.PositiveIntegerField(default=0)
    
    # Context management
    llm_cache_id = models.CharField(max_length=255, blank=True, null=True)
    
    class Meta:
        ordering = ['-last_activity']
        unique_together = ['user', 'dataset', 'is_active']

    def __str__(self):
        dataset_name = self.dataset.name if self.dataset else "General"
        return f"{self.user.username} - {dataset_name} ({self.created_at})"

class ChatMessage(models.Model):
    """Chat messages within analysis sessions."""
    MESSAGE_TYPES = [
        ('user', 'User Message'),
        ('ai', 'AI Response'),
        ('system', 'System Message'),
        ('analysis_result', 'Analysis Result'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(AnalysisSession, on_delete=models.CASCADE, related_name='messages')
    message_type = models.CharField(max_length=20, choices=MESSAGE_TYPES)
    content = models.TextField()
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['created_at']
    
    def __str__(self):
        return f"{self.session} - {self.message_type} - {self.created_at}"

    @property
    def metadata_json(self):
        """Helper for rendering JSON metadata in templates."""
        return json.dumps(self.metadata)

class AnalysisResult(models.Model):
    """
    Stores detailed analysis results linked to sessions.
    Hardened for production async execution.
    """
    class Status(models.TextChoices):
        PENDING = 'PENDING', 'Pending'
        RUNNING = 'RUNNING', 'Running'
        SUCCESS = 'SUCCESS', 'Success'
        FAILED = 'FAILED', 'Failed'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(AnalysisSession, on_delete=models.CASCADE, related_name='analysis_results')
    tool_used = models.CharField(max_length=100)
    query = models.TextField(blank=True, null=True) # User's original question
    
    # Payload
    result_data = models.JSONField(null=True, blank=True)
    ai_interpretation = models.TextField(blank=True, null=True)
    summary = models.TextField(blank=True, null=True) # For PPTX/Report generation
    
    # Async Observability
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.PENDING)
    task_id = models.CharField(max_length=255, blank=True, null=True)
    dedup_id = models.CharField(max_length=255, unique=True, blank=True, null=True)
    error_message = models.TextField(blank=True, null=True)
    
    # Timeline
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(blank=True, null=True)
    completed_at = models.DateTimeField(blank=True, null=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Analysis Result'
        verbose_name_plural = 'Analysis Results'
    
    def __str__(self):
        return f"Analysis for {self.tool_used} in session {self.session.id}"
    
class SiteSettings(models.Model):
    """
    Branding and global site configuration.
    """
    site_name = models.CharField(max_length=100, default='ChatFlow')
    site_description = models.TextField(blank=True, null=True)
    logo = models.ImageField(upload_to='branding/', blank=True, null=True)
    favicon = models.ImageField(upload_to='branding/', blank=True, null=True)
    contact_email = models.EmailField(blank=True, null=True)
    
    class Meta:
        verbose_name = 'Site Settings'
        verbose_name_plural = 'Site Settings'
    
    def __str__(self):
        return self.site_name

    def save(self, *args, **kwargs):
        """
        Ensure only one instance of SiteSettings exists (Singleton).
        """
        if not self.pk and SiteSettings.objects.exists():
            return # Pre-check prevent creation if one already exists
        return super(SiteSettings, self).save(*args, **kwargs)

    @classmethod
    def load(cls):
        """
        Load the current SiteSettings instance or create a default one.
        """
        obj, created = cls.objects.get_or_create(pk=1)
        return obj

class AIAuditLog(models.Model):
    """
    Enterprise-grade audit trail for AI interactions.
    Tracks prompts, responses, and costs (tokens).
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='ai_audit_logs', null=True, blank=True)
    session = models.ForeignKey(AnalysisSession, on_delete=models.SET_NULL, null=True, blank=True, related_name='audit_logs')
    prompt = models.TextField()
    response = models.TextField()
    
    # Cost & Usage metrics
    tokens_input = models.IntegerField(default=0)
    tokens_output = models.IntegerField(default=0)
    model_id = models.CharField(max_length=100)
    
    # Timeline
    timestamp = models.DateTimeField(auto_now_add=True)

    @property
    def tokens_total(self):
        return self.tokens_input + self.tokens_output
    
    class Meta:
        ordering = ['-timestamp']
        verbose_name = 'AI Audit Log'
        verbose_name_plural = 'AI Audit Logs'
    
    def __str__(self):
        return f"AI Audit - {self.model_id} - {self.timestamp}"

class UserQuota(models.Model):
    """
    Stores usage limits and plan information for a user.
    Enables future package system integration.
    """
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='quota')
    plan_name = models.CharField(max_length=50, default="Free")
    max_tokens = models.IntegerField(default=50000)
    current_tokens_used = models.IntegerField(default=0)
    expiry_date = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Quota: {self.user.email} ({self.plan_name})"

    @property
    def is_expired(self):
        if not self.expiry_date:
            return False
        from django.utils import timezone
        return timezone.now() > self.expiry_date

