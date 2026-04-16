import json
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from asgiref.sync import async_to_sync
from channels.testing import WebsocketCommunicator
from django.contrib.auth.models import AnonymousUser, User
from django.core.cache import cache
from django.db import IntegrityError
from django.test import Client, TestCase, TransactionTestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from alaina.asgi import application
from .agents.analysis_agent import AnalysisAgent
from .corporate_service import CorporateService
from .llm_management.gemini_service import GeminiService
from .models import (
    AIAuditLog,
    AnalysisResult,
    AnalysisSession,
    Corporate,
    CorporateInvitation,
    CreditPackage,
    Dataset,
    Invoice,
    UserQuota,
)


class SaihaSecurityTests(TestCase):
    def setUp(self):
        cache.clear()
        self.owner = User.objects.create_user(username='owner', email='owner@example.com', password='pw123456')
        self.other = User.objects.create_user(username='other', email='other@example.com', password='pw123456')
        self.dataset = Dataset.objects.create(
            user=self.owner,
            name='Sales',
            original_filename='sales.csv',
            file_type='csv',
            file_size=100,
            rows_count=10,
            columns_count=2,
            processed_file_path='datasets/owner/data.csv',
            metadata_file_path='datasets/owner/meta.json',
            is_processed=True,
        )
        self.session = AnalysisSession.objects.create(user=self.owner, dataset=self.dataset, session_name='Sales Session')
        self.client.force_login(self.other)

    def test_chat_analysis_rejects_foreign_session(self):
        response = self.client.post(
            reverse('saiha:api_chat_analysis'),
            data=json.dumps({'message': 'analyze', 'session_id': str(self.session.id)}),
            content_type='application/json',
        )

        self.assertEqual(response.status_code, 404)

    def test_delete_dataset_requires_post(self):
        self.client.force_login(self.owner)
        response = self.client.get(reverse('saiha:delete_dataset', args=[self.dataset.id]))
        self.assertEqual(response.status_code, 405)

    @override_settings(PHASE2_RATE_LIMITS={'chat_analysis': '2/m'})
    @patch('saiha.views.get_analysis_agent')
    def test_chat_analysis_is_rate_limited(self, mock_get_agent):
        self.client.force_login(self.owner)
        mock_get_agent.return_value = SimpleNamespace(process_query=lambda _message: ['task-1'])
        payload = json.dumps({'message': 'analyze', 'session_id': str(self.session.id)})

        first = self.client.post(reverse('saiha:api_chat_analysis'), data=payload, content_type='application/json')
        second = self.client.post(reverse('saiha:api_chat_analysis'), data=payload, content_type='application/json')
        third = self.client.post(reverse('saiha:api_chat_analysis'), data=payload, content_type='application/json')

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(third.status_code, 429)
        self.assertEqual(third.json()['scope'], 'chat_analysis')
        self.assertEqual(mock_get_agent.call_count, 2)


@override_settings(CHANNEL_LAYERS={'default': {'BACKEND': 'channels.layers.InMemoryChannelLayer'}})
class WebsocketAccessTests(TransactionTestCase):
    reset_sequences = True

    def setUp(self):
        cache.clear()
        self.owner = User.objects.create_user(username='wsowner', email='wsowner@example.com', password='pw123456')
        self.other = User.objects.create_user(username='wsother', email='wsother@example.com', password='pw123456')
        self.dataset = Dataset.objects.create(
            user=self.owner,
            name='Traffic',
            original_filename='traffic.csv',
            file_type='csv',
            file_size=100,
            rows_count=10,
            columns_count=2,
            processed_file_path='datasets/wsowner/data.csv',
            metadata_file_path='datasets/wsowner/meta.json',
            is_processed=True,
        )
        self.session = AnalysisSession.objects.create(user=self.owner, dataset=self.dataset, session_name='Traffic Session')

    def test_owner_can_connect_to_session_socket(self):
        async def scenario():
            communicator = WebsocketCommunicator(application, f"/ws/notifications/{self.session.id}/")
            communicator.scope['user'] = self.owner
            connected, _ = await communicator.connect()
            self.assertTrue(connected)
            await communicator.disconnect()

        async_to_sync(scenario)()

    def test_foreign_user_is_rejected_from_session_socket(self):
        async def scenario():
            communicator = WebsocketCommunicator(application, f"/ws/notifications/{self.session.id}/")
            communicator.scope['user'] = self.other
            connected, code = await communicator.connect()
            self.assertFalse(connected)
            self.assertEqual(code, 4003)

        async_to_sync(scenario)()

    def test_anonymous_user_is_rejected_from_session_socket(self):
        async def scenario():
            communicator = WebsocketCommunicator(application, f"/ws/notifications/{self.session.id}/")
            communicator.scope['user'] = AnonymousUser()
            connected, code = await communicator.connect()
            self.assertFalse(connected)
            self.assertEqual(code, 4001)

        async_to_sync(scenario)()


class CorporateInvitationTests(TestCase):
    def setUp(self):
        cache.clear()
        self.owner = User.objects.create_user(username='corp', email='corp@example.com', password='pw123456')
        self.other = User.objects.create_user(username='wrong', email='wrong@example.com', password='pw123456')
        self.corporate = Corporate.objects.create(name='Acme Corp', rem_credits=50, total_credits=50, max_users=5)
        self.invitation = CorporateInvitation.objects.create(
            corporate=self.corporate,
            email='invitee@example.com',
            token='12345678-1234-1234-1234-123456789abc',
            initial_credits=5,
            expires_at=timezone.now() + timedelta(days=1),
        )

    def test_join_rejects_email_mismatch_even_if_posted(self):
        self.client.force_login(self.other)
        response = self.client.post(reverse('saiha:corporate_join', args=[self.invitation.token]))
        self.invitation.refresh_from_db()

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'This invitation can only be accepted by the invited email address.')
        self.assertFalse(self.invitation.is_accepted)


class BillingAndUsageTests(TestCase):
    def setUp(self):
        cache.clear()
        self.user = User.objects.create_user(username='billing', email='billing@example.com', password='pw123456')
        self.package = CreditPackage.objects.create(
            name='Starter',
            credits=5,
            price_usd='5.00',
            price_bdt='600.00',
            is_active=True,
        )

    def test_user_topup_creates_invoice(self):
        self.client.force_login(self.user)
        response = self.client.post(reverse('saiha:user_topup'), {'package_id': self.package.id})
        payload = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload['status'], 'success')
        self.assertTrue(Invoice.objects.filter(user=self.user).exists())
        self.assertIsNotNone(payload['invoice_id'])

    def test_user_topup_requires_csrf(self):
        csrf_client = Client(enforce_csrf_checks=True)
        csrf_client.force_login(self.user)
        response = csrf_client.post(reverse('saiha:user_topup'), {'package_id': self.package.id})
        self.assertEqual(response.status_code, 403)

    def test_usage_kpi_total_comes_from_audit_logs(self):
        quota = self.user.quota
        quota.current_tokens_used = 1000
        quota.max_tokens = 10000
        quota.save()

        AIAuditLog.objects.create(
            user=self.user,
            prompt='p1',
            response='r1',
            tokens_input=10000,
            tokens_output=5000,
            model_id='test-model',
        )
        AIAuditLog.objects.create(
            user=self.user,
            prompt='p2',
            response='r2',
            tokens_input=2000,
            tokens_output=3000,
            model_id='test-model',
        )

        self.client.force_login(self.user)
        response = self.client.get(reverse('saiha:get_usage_data'))
        payload = response.json()

        self.assertEqual(payload['kpis']['total'], 2.0)

    def test_user_recharge_preserves_usage_history(self):
        quota = self.user.quota
        quota.max_tokens = 10000
        quota.current_tokens_used = 2500
        quota.save()

        CorporateService.recharge_user(self.user, 5, package=self.package)
        quota.refresh_from_db()

        self.assertEqual(quota.current_tokens_used, 2500)
        self.assertEqual(quota.max_tokens, 60000)

    @override_settings(AI_AUDIT_STORE_RAW_CONTENT=False)
    def test_ai_audit_logs_redact_raw_content_by_default(self):
        service = GeminiService.__new__(GeminiService)
        service.model_id = 'test-model'
        usage = SimpleNamespace(prompt_token_count=11, candidates_token_count=7, cached_content_token_count=0)

        service._log_interaction(
            prompt='sensitive prompt body',
            response_text='sensitive response body',
            usage=usage,
            user=self.user,
        )

        audit = AIAuditLog.objects.latest('timestamp')
        self.assertIn('[redacted prompt]', audit.prompt)
        self.assertIn('[redacted response]', audit.response)
        self.assertNotEqual(audit.prompt, 'sensitive prompt body')
        self.assertNotEqual(audit.response, 'sensitive response body')


class AnalysisSessionConstraintTests(TestCase):
    def setUp(self):
        cache.clear()
        self.user = User.objects.create_user(username='sessionuser', email='session@example.com', password='pw123456')
        self.dataset = Dataset.objects.create(
            user=self.user,
            name='Sessions',
            original_filename='sessions.csv',
            file_type='csv',
            file_size=100,
            rows_count=10,
            columns_count=2,
            processed_file_path='datasets/session/data.csv',
            metadata_file_path='datasets/session/meta.json',
            is_processed=True,
        )

    def test_only_one_active_session_is_allowed(self):
        AnalysisSession.objects.create(user=self.user, dataset=self.dataset, session_name='Active 1')
        with self.assertRaises(IntegrityError):
            AnalysisSession.objects.create(user=self.user, dataset=self.dataset, session_name='Active 2')

    def test_multiple_inactive_sessions_are_allowed(self):
        AnalysisSession.objects.create(user=self.user, dataset=self.dataset, session_name='Closed 1', is_active=False)
        AnalysisSession.objects.create(user=self.user, dataset=self.dataset, session_name='Closed 2', is_active=False)
        self.assertEqual(AnalysisSession.objects.filter(user=self.user, dataset=self.dataset, is_active=False).count(), 2)


class AnalysisGuardTests(TestCase):
    def setUp(self):
        cache.clear()
        self.user = User.objects.create_user(username='guarduser', email='guard@example.com', password='pw123456')
        self.dataset = Dataset.objects.create(
            user=self.user,
            name='Guard Dataset',
            original_filename='guard.csv',
            file_type='csv',
            file_size=100,
            rows_count=10,
            columns_count=2,
            processed_file_path='datasets/guard/data.csv',
            metadata_file_path='datasets/guard/meta.json',
            is_processed=True,
        )
        self.session = AnalysisSession.objects.create(user=self.user, dataset=self.dataset, session_name='Guard Session')

    def _dummy_tool(self):
        class DummyTool:
            def get_parameters_schema(self):
                return None

        return DummyTool()

    @override_settings(ANALYSIS_MAX_TOOLS_PER_REQUEST=3, ANALYSIS_SESSION_COOLDOWN_SECONDS=0)
    @patch('saiha.agents.analysis_agent.send_ws_notification')
    @patch('saiha.agents.analysis_agent.execute_analysis_task.apply_async')
    @patch('saiha.agents.analysis_agent.tool_registry.get_tool')
    @patch('saiha.agents.analysis_agent.analysis_planner.create_plan')
    def test_process_query_caps_tool_fanout(self, mock_create_plan, mock_get_tool, mock_apply_async, mock_notify):
        mock_create_plan.return_value = [
            {'tool': 'tool_a', 'params': {}},
            {'tool': 'tool_b', 'params': {}},
            {'tool': 'tool_c', 'params': {}},
            {'tool': 'tool_d', 'params': {}},
        ]
        mock_get_tool.return_value = self._dummy_tool()
        mock_apply_async.side_effect = [
            SimpleNamespace(id='task-1'),
            SimpleNamespace(id='task-2'),
            SimpleNamespace(id='task-3'),
        ]

        agent = AnalysisAgent(self.session)
        task_ids = agent.process_query('run many tools')

        self.assertEqual(task_ids, ['task-1', 'task-2', 'task-3'])
        self.assertEqual(mock_apply_async.call_count, 3)
        self.assertTrue(any('can only run 3 at a time' in call.args[0] for call in mock_notify.call_args_list if call.args))

    @override_settings(ANALYSIS_MAX_ACTIVE_TASKS_PER_SESSION=2, ANALYSIS_SESSION_COOLDOWN_SECONDS=0)
    @patch('saiha.agents.analysis_agent.send_ws_notification')
    @patch('saiha.agents.analysis_agent.execute_analysis_task.apply_async')
    @patch('saiha.agents.analysis_agent.analysis_planner.create_plan')
    def test_process_query_blocks_when_too_many_tasks_are_active(self, mock_create_plan, mock_apply_async, mock_notify):
        AnalysisResult.objects.create(session=self.session, tool_used='tool_a', status=AnalysisResult.Status.PENDING, dedup_id='pending-1')
        AnalysisResult.objects.create(session=self.session, tool_used='tool_b', status=AnalysisResult.Status.RUNNING, dedup_id='running-1')

        agent = AnalysisAgent(self.session)
        task_ids = agent.process_query('run another')

        self.assertEqual(task_ids, [])
        self.assertEqual(mock_create_plan.call_count, 0)
        self.assertEqual(mock_apply_async.call_count, 0)
        self.assertTrue(any('too many analyses in progress' in call.args[0] for call in mock_notify.call_args_list if call.args))

    @override_settings(ANALYSIS_SESSION_COOLDOWN_SECONDS=60)
    @patch('saiha.agents.analysis_agent.send_ws_notification')
    @patch('saiha.agents.analysis_agent.execute_analysis_task.apply_async')
    @patch('saiha.agents.analysis_agent.tool_registry.get_tool')
    @patch('saiha.agents.analysis_agent.analysis_planner.create_plan')
    def test_process_query_enforces_session_cooldown(self, mock_create_plan, mock_get_tool, mock_apply_async, mock_notify):
        mock_create_plan.return_value = [{'tool': 'tool_a', 'params': {}}]
        mock_get_tool.return_value = self._dummy_tool()
        mock_apply_async.return_value = SimpleNamespace(id='task-1')

        agent = AnalysisAgent(self.session)
        first = agent.process_query('first')
        second = agent.process_query('second')

        self.assertEqual(first, ['task-1'])
        self.assertEqual(second, [])
        self.assertEqual(mock_apply_async.call_count, 1)
        self.assertTrue(any('Please wait 60 seconds' in call.args[0] for call in mock_notify.call_args_list if call.args))


class InterpretationDeliveryTests(TestCase):
    def setUp(self):
        cache.clear()
        self.user = User.objects.create_user(username='interpuser', email='interp@example.com', password='pw123456')
        self.dataset = Dataset.objects.create(
            user=self.user,
            name='Interpret Dataset',
            original_filename='interpret.csv',
            file_type='csv',
            file_size=100,
            rows_count=10,
            columns_count=2,
            processed_file_path='datasets/interpret/data.csv',
            metadata_file_path='datasets/interpret/meta.json',
            is_processed=True,
        )
        self.session = AnalysisSession.objects.create(user=self.user, dataset=self.dataset, session_name='Interpret Session')
        self.result = AnalysisResult.objects.create(
            session=self.session,
            tool_used='dataset_overview',
            query='give me an overview',
            status=AnalysisResult.Status.SUCCESS,
            result_data={
                'data': {'rows': 10, 'columns': 2},
                'artifacts': [{'type': 'table', 'title': 'Overview', 'headers': ['Metric', 'Value'], 'rows': [['Rows', 10], ['Columns', 2]]}],
                'message': 'Dataset overview complete.'
            },
            dedup_id='interpret-result-1',
        )

    @patch('saiha.agents.memory_manager.MemoryManager.update_summary', side_effect=NameError("name 'session' is not defined"))
    @patch('saiha.agents.memory_manager.MemoryManager.update_working_memory')
    @patch('saiha.agents.memory_manager.MemoryManager.decay_stale_state')
    @patch('saiha.agents.interpretation_agent.gemini_service.generate_content')
    @patch('saiha.agents.interpretation_agent.gemini_service.get_or_create_cache')
    @patch('channels.layers.get_channel_layer')
    @patch('asgiref.sync.async_to_sync')
    def test_interpretation_still_persists_and_broadcasts_when_memory_update_fails(
        self,
        mock_async_to_sync,
        mock_get_channel_layer,
        mock_get_cache,
        mock_generate_content,
        _mock_decay,
        _mock_update_memory,
        _mock_update_summary,
    ):
        from .agents.interpretation_agent import InterpretationAgent

        mock_get_cache.return_value = None
        mock_generate_content.return_value = (
            "Key takeaway text.\n\n[METADATA]\n"
            "type: overview\n"
            "target: N/A\n"
            "columns: Order ID|Sales"
        )
        group_send = MagicMock()
        mock_async_to_sync.side_effect = lambda fn: fn
        mock_get_channel_layer.return_value = SimpleNamespace(group_send=group_send)

        output = InterpretationAgent().interpret_result(str(self.result.id))

        self.assertIn('Key takeaway text.', output)
        self.result.refresh_from_db()
        self.assertIn('Key takeaway text.', self.result.ai_interpretation)
        self.assertTrue(self.session.messages.filter(message_type='ai', content__icontains='Key takeaway text.').exists())
        self.assertEqual(group_send.call_count, 1)
