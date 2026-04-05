import json
import os
import logging
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse, HttpResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.core.files.storage import default_storage
from django.conf import settings

from .models import Dataset, DatasetColumn, AnalysisSession, ChatMessage, AnalysisResult
from .database_processing_logic.dataset_processor import DatasetProcessor
from .database_processing_logic.storage_manager_parquet import DatasetStorageManager
from .session_management.session_manager import SessionManager
from .agents.analysis_agent import get_analysis_agent
from .reporting.report_builder import ReportBuilder
from .reporting.pptx_exporter import PPTXExporter
from .reporting.docx_exporter import DOCXExporter

logger = logging.getLogger(__name__)

@login_required
def index(request):
    """
    Renders the index page with active datasets and chat sessions.
    Resolves the current session based on dataset_id or session_id.
    """
    dataset_id = request.GET.get('dataset_id')
    session_id = request.GET.get('session_id')
    
    current_session = None
    messages = []
    
    if session_id:
        current_session = AnalysisSession.objects.filter(id=session_id, user=request.user).first()
    elif dataset_id:
        # Resolve or create session for selected dataset
        current_session = SessionManager.get_or_create_session(request.user, dataset_id)
    
    if not current_session:
        # Fallback: Get most recent active session
        current_session = AnalysisSession.objects.filter(
            user=request.user, 
            is_active=True
        ).order_by('-last_activity').first()

    if current_session:
        messages = SessionManager.get_history(current_session)

    datasets = Dataset.objects.filter(user=request.user, is_active=True).order_by('-upload_date')
    chat_sessions = AnalysisSession.objects.filter(user=request.user, is_active=True).order_by('-last_activity')
    
    context = {
        'chat_sessions': chat_sessions,
        'current_session': current_session,
        'messages': messages,
        'available_datasets': datasets,
    }
    return render(request, 'index.html', context)

@csrf_exempt
@login_required
def upload_dataset(request):
    """
    Processes uploaded dataset, saves it as Parquet, and creates DB records.
    """
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        
        processor = DatasetProcessor()
        storage_manager = DatasetStorageManager()
        
        try:
            # 1. Process and Clean File
            df, metadata = processor.process_file(uploaded_file)
            
            # 2. Create Dataset record placeholder to get UUID
            dataset = Dataset.objects.create(
                user=request.user,
                name=uploaded_file.name,
                original_filename=uploaded_file.name,
                file_type=metadata['file_type'],
                storage_format='parquet',
                file_size=uploaded_file.size,
                rows_count=metadata['rows_count'],
                columns_count=metadata['columns_count'],
                is_processed=True
            )
            
            # 3. Save as Parquet
            file_path = storage_manager.save_processed_file(df, request.user.id, dataset.id, metadata['file_type'])
            
            # 4. Save Metadata & Preview
            col_metadata = processor.get_column_metadata(df)
            metadata_path = storage_manager.save_metadata({'columns': col_metadata}, request.user.id, dataset.id, df=df)
            
            # 5. Update paths in DB
            dataset.processed_file_path = storage_manager.get_relative_path(file_path)
            dataset.metadata_file_path = storage_manager.get_relative_path(metadata_path)
            dataset.save()
            
            # 6. Create Column records
            for col in col_metadata:
                DatasetColumn.objects.create(
                    dataset=dataset,
                    column_name=col['column_name'],
                    column_index=col['column_index'],
                    data_type=col['data_type'],
                    null_count=col['null_count'],
                    unique_count=col['unique_count'],
                    sample_values=col['sample_values']
                )
            
            # 7. Automatically create and initialize an analysis session
            session = SessionManager.get_or_create_session(request.user, dataset.id)
            
            return JsonResponse({
                'status': 'success', 
                'dataset_id': str(dataset.id), 
                'session_id': str(session.id) if session else None,
                'name': dataset.name,
                'rows': dataset.rows_count,
                'cols': dataset.columns_count
            })
            
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
            
    return JsonResponse({'status': 'error', 'message': 'Invalid request.'}, status=405)

@csrf_exempt
@login_required
def api_chat_analysis(request):
    """
    The Centralized Analytical Chat API for ChatFlow.
    Dispatched via the AnalysisAgent.
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            message = data.get('message', '')
            session_id = data.get('session_id')
            
            if not session_id:
                return JsonResponse({'status': 'error', 'message': 'Session ID is required.'}, status=400)
            
            # 1. Initialize the AnalysisAgent
            agent = get_analysis_agent(session_id)
            
            # 2. Process Query (Async Pipeline)
            task_ids = agent.process_query(message)
            
            return JsonResponse({
                'status': 'success',
                'session_id': session_id,
                'tasks': task_ids,
                'message': 'Analysis request received and dispatched.'
            })
        except Exception as e:
            logger.error(f"Chat API Error: {e}", exc_info=True)
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'POST required'}, status=405)

@login_required
def update_profile(request):
    """
    Updates the user's profile information (Display Name).
    """
    if request.method == 'POST':
        try:
            display_name = request.POST.get('display_name', '').strip()
            if display_name:
                user = request.user
                user.first_name = display_name
                user.save()
                return JsonResponse({
                    'status': 'success', 
                    'message': 'Profile updated successfully!', 
                    'display_name': display_name,
                    'initials': display_name[0].upper() if display_name else ''
                })
            return JsonResponse({'status': 'error', 'message': 'Name cannot be empty.'}, status=400)
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    return JsonResponse({'error': 'POST required'}, status=405)

@login_required
def dataset_dashboard(request):
    """
    Renders the dedicated Dataset Management Dashboard.
    """
    datasets = Dataset.objects.filter(user=request.user, is_active=True).order_by('-upload_date')
    return render(request, 'datasets.html', {'datasets': datasets})

@csrf_exempt
@login_required
def delete_dataset(request, dataset_id):
    """
    Soft-deletes a dataset via HTMX or standard request.
    """
    dataset = get_object_or_404(Dataset, id=dataset_id, user=request.user)
    dataset.is_active = False
    dataset.save()
    
    if request.headers.get('HX-Request'):
        # Return empty response to remove element via HTMX
        return HttpResponse("") 
        
@login_required
def dataset_detail(request, dataset_id):
    """
    Shows detailed schema, lineage, and data preview for a specific dataset.
    """
    dataset = get_object_or_404(Dataset, id=dataset_id, user=request.user)
    storage_manager = DatasetStorageManager()
    
    # 1. Load Preview Data (Top 10 rows)
    try:
        df = storage_manager.load_processed_file(request.user.id, dataset.id)
        preview_data = df.head(10).to_dict('records')
        columns = df.columns.tolist()
    except Exception:
        preview_data = []
        columns = []
        
    # 2. Get Lineage
    parent = dataset.parent_dataset
    children = dataset.derived_datasets.filter(is_active=True)
    
    context = {
        'dataset': dataset,
        'columns': dataset.columns.all().order_by('column_index'),
        'preview_columns': columns,
        'preview_data': preview_data,
        'parent': parent,
        'children': children,
    }
    return render(request, 'dataset_detail.html', context)
@login_required
def get_analysis_result(request, result_id):
    """
    API endpoint for lazy loading detailed analysis results (like large charts).
    """
    result = get_object_or_404(AnalysisResult, id=result_id, session__user=request.user)
    return JsonResponse({
        'id': str(result.id),
        'tool': result.tool_used,
        'data': result.result_data,
        'interpretation': result.ai_interpretation,
        'created_at': result.created_at.isoformat()
    })

@login_required
def export_session_report(request, session_id, format):
    """
    The Consulting-Grade Export Endpoint.
    Orchestrates the Narrative Builder -> Selection -> Professional Exporter.
    """
    session = get_object_or_404(AnalysisSession, id=session_id, user=request.user)
    
    # 1. Build Narrative Intelligence (Filtering, Summarizing, Title Generation)
    builder = ReportBuilder(session_id=str(session.id))
    report_context = builder.build_narrative_context()
    
    # 2. Dispatch to professional layout engine
    if format.lower() == 'pptx':
        exporter = PPTXExporter()
        file_stream = exporter.generate_report(report_context)
        content_type = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        filename = f"Analytical_Report_{session.dataset.name}.pptx"
    elif format.lower() == 'docx':
        exporter = DOCXExporter()
        file_stream = exporter.generate_report(report_context)
        content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        filename = f"Narrative_Report_{session.dataset.name}.docx"
    else:
        return HttpResponse("Unsupported Format", status=400)
    
    return FileResponse(
        file_stream,
        as_attachment=True,
        filename=filename,
        content_type=content_type
    )
