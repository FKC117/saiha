import json
import os
import logging
import datetime
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse, HttpResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.core.files.storage import default_storage
from django.conf import settings

from sys import path
from django.utils import timezone
from django.contrib.auth.models import User
from saiha.models import (
    Dataset, DatasetColumn, AnalysisSession, ChatMessage, AnalysisResult,
    AIAuditLog, UserQuota, AppConfiguration, Corporate, CorporateProfile, CorporateInvitation, CreditPackage
)
from saiha.corporate_service import CorporateService
from saiha.adapter import CustomAccountAdapter
from saiha.database_processing_logic.dataset_processor import DatasetProcessor, EmptyColumnsDetected
from saiha.database_processing_logic.storage_manager_parquet import DatasetStorageManager
from saiha.session_management.session_manager import SessionManager
from saiha.agents.analysis_agent import get_analysis_agent
from saiha.reporting.report_builder import ReportBuilder
from saiha.reporting.pptx_exporter import PPTXExporter
from saiha.reporting.docx_exporter import DOCXExporter

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
    credit_packages = CreditPackage.objects.filter(is_active=True)
    
    context = {
        'chat_sessions': chat_sessions,
        'current_session': current_session,
        'messages': messages,
        'available_datasets': datasets,
        'credit_packages': credit_packages,
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
            drop_empty = request.POST.get('drop_empty') == 'true'
            # 1. Process and Clean File
            df, metadata = processor.process_file(uploaded_file, drop_empty=drop_empty)
            
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
            
        except EmptyColumnsDetected as e:
            return JsonResponse({
                'status': 'warning', 
                'message': str(e),
                'empty_columns': e.columns
            }, status=400)
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
            from saiha.celery_tasks.analysis_tasks import send_ws_notification
            logging.getLogger(__name__).error(f"Chat API Error: {e}", exc_info=True)
            
            # BROADCAST ERROR TO UI TO STOP LOADER
            if session_id:
                send_ws_notification(
                    f"An error occurred while processing your request: {str(e)}",
                    status="error",
                    session_id=str(session_id)
                )
            
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

@login_required
def get_usage_data(request):
    """
    API endpoint to fetch token usage statistics for the current user.
    Aggregates data for charts and KPI cards.
    """
    from django.db.models import Sum
    from django.db.models.functions import TruncDate
    from datetime import timedelta
    
    # 0. Sync credits first
    CorporateService.sync_user_credits(request.user)

    user = request.user
    today = timezone.now().date()
    fourteen_days_ago = today - timedelta(days=14)

    # 1. Quota Info
    quota, _ = UserQuota.objects.get_or_create(user=user)
    
    # 2. Daily Usage (Last 14 days)
    daily_stats = AIAuditLog.objects.filter(
        user=user, 
        timestamp__date__gte=fourteen_days_ago
    ).annotate(
        date=TruncDate('timestamp')
    ).values('date').annotate(
        total=Sum('tokens_input') + Sum('tokens_output')
    ).order_by('date')

    # Format for chart
    dates = []
    usage_values = []
    
    # Fill in gaps with zeros
    stats_dict = {s['date']: s['total'] for s in daily_stats}
    current_date = fourteen_days_ago
    while current_date <= today:
        dates.append(current_date.strftime('%b %d'))
        usage_values.append(stats_dict.get(current_date, 0))
        current_date += timedelta(days=1)

    # 3. Top Sessions
    top_sessions = AIAuditLog.objects.filter(user=user, session__isnull=False).values(
        'session__id', 'session__session_name', 'session__dataset__name'
    ).annotate(
        total=Sum('tokens_input') + Sum('tokens_output')
    ).order_by('-total')[:5]

    formatted_sessions = []
    for s in top_sessions:
        name = s['session__session_name'] or s['session__dataset__name'] or "Unnamed Session"
        formatted_sessions.append({'name': name, 'value': s['total']})

    # 4. Totals
    today_total_tokens = AIAuditLog.objects.filter(
        user=user, 
        timestamp__date=today
    ).aggregate(
        total=Sum('tokens_input') + Sum('tokens_output')
    )['total'] or 0

    # Get Dynamic Conversion Rate
    rate = float(AppConfiguration.get_rate()) or 10000.0

    return JsonResponse({
        'status': 'success',
        'plan_name': quota.plan_name,
        'used_tokens': quota.current_tokens_used,
        'max_tokens': quota.max_tokens,
        'rescue_tokens': quota.expired_tokens,
        'expiry_date': quota.expiry_date.isoformat() if quota.expiry_date else None,
        'is_expired': quota.is_expired,
        'kpis': {
            'today': round(today_total_tokens / rate, 3), 
            'total': quota.credits_used
        },
        'charts': {
            'daily_dates': dates,
            'daily_values': [round(v / rate, 2) for v in usage_values],
            'sessions': [{'name': s['name'], 'value': round(s['value'] / rate, 2)} for s in formatted_sessions]
        }
    })

@csrf_exempt
@login_required
def user_topup(request):
    """
    API to process retail user credit top-ups.
    """
    if request.method == 'POST':
        package_id = request.POST.get('package_id')
        try:
            package = get_object_or_404(CreditPackage, id=package_id, is_active=True)
            amount = float(package.credits)
            CorporateService.recharge_user(request.user, amount)
            return JsonResponse({
                'status': 'success', 
                'message': f'Payment Successful! {package.name} ({amount} Credits) added. Any expired balance has been rescued.'
            })
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
    return JsonResponse({'status': 'error', 'message': 'Invalid request.'})

# --- CORPORATE ADMIN PANEL VIEWS ---

from functools import wraps
from django.core.exceptions import PermissionDenied

def corporate_admin_required(view_func):
    @wraps(view_func)
    @login_required
    def _wrapped_view(request, *args, **kwargs):
        # Check if user has a CorporateProfile with Role.ADMIN
        profile = getattr(request.user, 'corp_profile', None)
        if not profile or profile.role != CorporateProfile.Role.ADMIN:
            raise PermissionDenied("Access to Corporate Dashboard is restricted to Administrators.")
        return view_func(request, *args, **kwargs)
    return _wrapped_view

@corporate_admin_required
def corporate_dashboard(request):
    """
    Main management portal for Corporate Admins.
    """
    corporate = request.user.corp_profile.corporate
    
    # 1. Sync Expiry and Rollover
    corporate = CorporateService.sync_corporate_credits(corporate)
    
    members = CorporateProfile.objects.filter(corporate=corporate, is_active=True).select_related('user', 'user__quota')
    pending_invites = CorporateInvitation.objects.filter(corporate=corporate, is_accepted=False).order_by('-created_at')
    
    from django.db.models import Sum
    
    rate = float(AppConfiguration.get_rate())
    
    # Calculate isolated organization usage
    total_spent = 0
    member_stats = []
    
    # 1. Active Members
    for m in members:
        # Sum only logs that happened AFTER they joined this corp
        usage_tokens = AIAuditLog.objects.filter(
            user=m.user,
            timestamp__gte=m.joined_at
        ).aggregate(
            total=Sum('tokens_input') + Sum('tokens_output')
        )['total'] or 0
        
        usage_credits = round(usage_tokens / rate, 2)
        total_spent += usage_credits
        
        member_stats.append({
            'profile': m,
            'org_usage': usage_credits,
            'total_allocation': m.user.quota.max_credits
        })
    
    # 2. Former Members (Historical consumption & Archive UI)
    former_members = CorporateProfile.objects.filter(corporate=corporate, is_active=False).select_related('user')
    former_member_stats = []
    
    for fm in former_members:
        # Sum logs within their tenure window
        usage_tokens = AIAuditLog.objects.filter(
            user=fm.user,
            timestamp__gte=fm.joined_at,
            timestamp__lte=fm.left_at
        ).aggregate(
            total=Sum('tokens_input') + Sum('tokens_output')
        )['total'] or 0
        
        usage_credits = round(usage_tokens / rate, 2)
        total_spent += usage_credits # Still count in total
        
        former_member_stats.append({
            'profile': fm,
            'tenure_usage': usage_credits
        })
    
    # 3. Invitations
    pending_invites = CorporateInvitation.objects.filter(corporate=corporate, is_accepted=False, expires_at__gt=timezone.now())

    # 4. Purchase Packages
    credit_packages = CreditPackage.objects.filter(is_active=True)

    context = {
        'corporate': corporate,
        'member_stats': member_stats,
        'former_member_stats': former_member_stats,
        'pending_invites': pending_invites,
        'credit_packages': credit_packages,
        'total_spent': round(total_spent, 2),
        'unallocated': corporate.rem_credits,
        'expired_credits': corporate.expired_credits,
        'expiry_date': corporate.expiry_date,
        'active_seats': members.filter(role='MEMBER').count()
    }
    return render(request, 'corporate/dashboard.html', context)

@csrf_exempt
@corporate_admin_required
def corporate_remove_member(request):
    """
    API to discontinue a member.
    """
    if request.method == 'POST':
        user_id = request.POST.get('user_id')
        user = get_object_or_404(User, id=user_id)
        corporate = request.user.corp_profile.corporate
        
        try:
            CorporateService.discontinue_member(corporate, user)
            return JsonResponse({'status': 'success', 'message': f'Member {user.email} discontinued successfully.'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})

@csrf_exempt
@corporate_admin_required
def corporate_resend_invite(request):
    """
    API to resend an invitation.
    """
    if request.method == 'POST':
        invite_id = request.POST.get('invite_id')
        try:
            CorporateService.resend_invitation(request, invite_id)
            return JsonResponse({'status': 'success', 'message': 'Invitation resent successfully.'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})

@csrf_exempt
@corporate_admin_required
def corporate_topup(request):
    """
    API to process credit top-ups using dynamic packages.
    """
    if request.method == 'POST':
        package_id = request.POST.get('package_id')
        corporate = request.user.corp_profile.corporate
        
        try:
            package = get_object_or_404(CreditPackage, id=package_id, is_active=True)
            amount = float(package.credits)
            
            total_added = CorporateService.recharge_corporate(corporate, amount)
            return JsonResponse({
                'status': 'success', 
                'message': f'Payment Successful! {package.name} ({amount} Credits) added. Any expired balance has been carried forward.'
            })
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
            
    return JsonResponse({'status': 'error', 'message': 'Invalid request.'})

@csrf_exempt
@corporate_admin_required
def corporate_purchase_seats(request):
    """
    API to purchase more seats using corporate credits.
    """
    if request.method == 'POST':
        count = int(request.POST.get('count', 0))
        corporate = request.user.corp_profile.corporate
        
        try:
            CorporateService.purchase_seats(corporate, count)
            return JsonResponse({'status': 'success', 'message': f'Successfully added {count} seats to your organization.'})
        except ValueError as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': f"Internal Error: {str(e)}"})
            
    return JsonResponse({'status': 'error', 'message': 'Invalid request.'})

@csrf_exempt
@corporate_admin_required
def corporate_add_member(request):
    """
    API to add a member to the corporate pool.
    """
    if request.method == 'POST':
        email = request.POST.get('email')
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        corporate = request.user.corp_profile.corporate
        
        try:
            # Check if user exists
            user = User.objects.filter(email=email).first()
            if not user:
                # Store invitation
                invitation = CorporateService.invite_user(corporate, email, first_name=first_name, last_name=last_name)
                # Send the branded email
                CorporateService.send_invitation_email(request, invitation)
                
                return JsonResponse({
                    'status': 'success', 
                    'message': f'Invitation sent to {email}.'
                })
            
            CorporateService.add_user_directly(corporate, user, first_name=first_name, last_name=last_name)
            return JsonResponse({'status': 'success', 'message': f'Member {email} added successfully.'})
        except ValueError as e:
            return JsonResponse({'status': 'error', 'message': str(e)})

    return JsonResponse({'status': 'error', 'message': 'Invalid request.'})

@csrf_exempt
@corporate_admin_required
def corporate_reallocate_credits(request):
    """
    API to change a member's credit limit.
    """
    if request.method == 'POST':
        user_id = request.POST.get('user_id')
        new_limit = float(request.POST.get('credits', 0))
        corporate = request.user.corp_profile.corporate
        
        try:
            target_user = get_object_or_404(User, id=user_id)
            CorporateService.reallocate_credits(corporate, target_user, new_limit)
            return JsonResponse({'status': 'success', 'message': 'Credits updated.'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
            
    return JsonResponse({'status': 'error', 'message': 'Invalid request.'})

from allauth.account.forms import LoginForm
from django.contrib.auth import login as auth_login

def corporate_login(request):
    """
    Dedicated login view for the Corporate Portal.
    Restricts access to users with a CorporateProfile.
    """
    if request.user.is_authenticated:
        return redirect(CustomAccountAdapter(request).get_login_redirect_url(request))

    form = LoginForm(data=request.POST or None, request=request)
    if request.method == "POST" and form.is_valid():
        user = form.user
        # High-security check: Only users with a CorporateProfile allowed here
        if not getattr(user, 'corp_profile', None):
            return render(request, 'corporate/login.html', {
                'form': form,
                'error': 'Access Restricted. This portal is for Corporate members only.'
            })
        
        # Log the user in
        auth_login(request, user)
        
        # Use our adapter for smart redirection
        return redirect(CustomAccountAdapter(request).get_login_redirect_url(request))

    return render(request, 'corporate/login.html', {'form': form})

def corporate_join(request, token):
    """
    Landing page for users accepting an invitation to join a corporation.
    """
    invitation = get_object_or_404(CorporateInvitation, id=token, is_accepted=False)
    
    if invitation.expires_at < timezone.now():
        return render(request, 'corporate/join.html', {'error': 'This invitation has expired.'})

    if request.method == 'POST':
        action = request.POST.get('action')
        
        if not request.user.is_authenticated:
            # Store invitation token in session so we can link it after they signup/login
            request.session['pending_invite_token'] = str(token)
            return redirect('account_signup')
        
        # Identity Guard: Check if logged-in user matches invitation email
        email_mismatch = request.user.email.lower() != invitation.email.lower()
        
        if email_mismatch and action != 'force_link':
            # Redirect to render the landing page with mismatch warning
            return render(request, 'corporate/join.html', {
                'invitation': invitation,
                'email_mismatch': True,
                'current_email': request.user.email
            })

        # If already logged in and (emails match OR user forced link)
        try:
            CorporateService.add_user_directly(
                invitation.corporate, 
                request.user, 
                initial_credits=invitation.initial_credits,
                first_name=invitation.first_name,
                last_name=invitation.last_name
            )
            invitation.is_accepted = True
            invitation.status = CorporateInvitation.Status.ACCEPTED
            invitation.save()
            return redirect('corporate_dashboard')
        except Exception as e:
            return render(request, 'corporate/join.html', {'invitation': invitation, 'error': str(e)})

    # GET request
    context = {'invitation': invitation}
    if request.user.is_authenticated:
        context['email_mismatch'] = request.user.email.lower() != invitation.email.lower()
        context['current_email'] = request.user.email
        
    return render(request, 'corporate/join.html', context)

@corporate_admin_required
def corporate_analytics(request):
    """
    Renders the Corporate-wide usage analytics dashboard.
    """
    corporate = request.user.corp_profile.corporate
    
    from django.db.models import Sum
    members = CorporateProfile.objects.filter(corporate=corporate, is_active=True)
    rate = float(AppConfiguration.get_rate())
    
    total_spent = 0
    for m in members:
        usage_tokens = AIAuditLog.objects.filter(
            user=m.user,
            timestamp__gte=m.joined_at
        ).aggregate(total=Sum('tokens_input') + Sum('tokens_output'))['total'] or 0
        total_spent += (usage_tokens / rate)
    
    former_members = CorporateProfile.objects.filter(corporate=corporate, is_active=False)
    for fm in former_members:
        usage_tokens = AIAuditLog.objects.filter(
            user=fm.user,
            timestamp__gte=fm.joined_at,
            timestamp__lte=fm.left_at
        ).aggregate(total=Sum('tokens_input') + Sum('tokens_output'))['total'] or 0
        total_spent += (usage_tokens / rate)

    context = {
        'corporate': corporate,
        'total_spent': round(total_spent, 2)
    }
    return render(request, 'corporate/analytics.html', context)

@corporate_admin_required
def get_corporate_usage_data(request):
    """
    API endpoint that aggregates usage for the entire organization with 30-day backfilling.
    """
    from django.db.models import Sum
    from django.db.models.functions import TruncDate
    from datetime import timedelta
    
    corporate = request.user.corp_profile.corporate
    
    # 1. Sync Expiry and Rollover
    corporate = CorporateService.sync_corporate_credits(corporate)
    
    # 2. Member Statistics for Charts
    members = CorporateProfile.objects.filter(corporate=corporate)
    
    today = timezone.now().date()
    thirty_days_ago = today - timedelta(days=30)
    
    daily_stats = {}
    member_stats_map = {}

    for profile in members:
        # Sum logs only for their tenure
        query = AIAuditLog.objects.filter(user=profile.user, timestamp__date__gte=thirty_days_ago)
        if profile.is_active:
            query = query.filter(timestamp__gte=profile.joined_at)
        else:
            query = query.filter(timestamp__gte=profile.joined_at, timestamp__lte=profile.left_at)
            
        logs = query.annotate(date=TruncDate('timestamp')).values('date').annotate(
            day_total=Sum('tokens_input') + Sum('tokens_output')
        )

        for log in logs:
            dt_str = log['date'].strftime('%Y-%m-%d')
            daily_stats[dt_str] = daily_stats.get(dt_str, 0) + log['day_total']
        
        # Calculate total for pie chart
        total_tokens = query.aggregate(total=Sum('tokens_input') + Sum('tokens_output'))['total'] or 0
        member_stats_map[profile.user.username] = total_tokens

    # BACKFILL: Ensure every day in the last 30 days exists for ECharts
    rate = float(AppConfiguration.get_rate())
    usage_dates = []
    usage_values = []
    
    for i in range(31):
        dt = thirty_days_ago + timedelta(days=i)
        dt_str = dt.strftime('%Y-%m-%d')
        usage_dates.append(dt_str)
        tokens = daily_stats.get(dt_str, 0)
        usage_values.append(round(tokens / rate, 2))

    final_breakdown = [
        {'name': name, 'value': round(tokens / rate, 2)}
        for name, tokens in member_stats_map.items()
    ]

    return JsonResponse({
        'status': 'success',
        'charts': {
            'daily_dates': usage_dates,
            'daily_values': usage_values,
            'member_breakdown': final_breakdown
        }
    })
