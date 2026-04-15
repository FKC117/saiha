# Review Findings

This file captures the highest-signal bugs, loopholes, and implementation risks found during a code review of the project.

## Critical

### 1. Cross-account analysis access via unowned session IDs

- Severity: Critical
- Files: [saiha/views.py](/F:/saiha/alaina/saiha/views.py:156), [saiha/agents/analysis_agent.py](/F:/saiha/alaina/saiha/agents/analysis_agent.py:19)
- Problem:
  `api_chat_analysis` accepts any `session_id` from the client, then calls `get_analysis_agent(session_id)` with no ownership check. `AnalysisAgent.__init__` loads `AnalysisSession` by raw ID only.
- Impact:
  Any authenticated user who learns another session UUID can trigger analysis runs against that session and its dataset, polluting another user’s history and consuming their credits/tokens.
- Recommended fix:
  Resolve the session inside the view with `get_object_or_404(AnalysisSession, id=session_id, user=request.user)` and pass the session object, not a raw ID, into the agent.

### 2. WebSocket notification stream can be subscribed to without authorization

- Severity: Critical
- Files: [saiha/consumers.py](/F:/saiha/alaina/saiha/consumers.py:10), [saiha/routing.py](/F:/saiha/alaina/saiha/routing.py:5)
- Problem:
  `NotificationConsumer.connect()` accepts any `session_id` and joins `notification_<session_id>` with no auth or ownership validation.
- Impact:
  Anyone who knows or guesses a session UUID can receive another user’s analysis progress, interpretations, and result metadata over WebSockets.
- Recommended fix:
  Reject unauthenticated users and verify that the connected user owns the `AnalysisSession` before joining the group.

### 3. Invitation flow can attach the wrong logged-in account to a corporate org

- Severity: Critical
- Files: [saiha/views.py](/F:/saiha/alaina/saiha/views.py:807), [saiha/corporate_service.py](/F:/saiha/alaina/saiha/corporate_service.py:151)
- Problem:
  `corporate_join` allows `action=force_link` when the logged-in email does not match the invited email. The code then directly adds the current account to the corporate org and marks the invitation accepted.
- Impact:
  A forwarded invitation link can be redeemed by the wrong account, granting unauthorized corporate membership and credit allocation.
- Recommended fix:
  Remove the force-link bypass, lookup invitations by the dedicated `token` field, and require the authenticated account email to match the invitation email unless an admin explicitly reissues the invite.

## High

### 4. Authenticated money/credit mutation endpoints are CSRF-exempt

- Severity: High
- Files: [saiha/views.py](/F:/saiha/alaina/saiha/views.py:79), [saiha/views.py](/F:/saiha/alaina/saiha/views.py:156), [saiha/views.py](/F:/saiha/alaina/saiha/views.py:230), [saiha/views.py](/F:/saiha/alaina/saiha/views.py:410), [saiha/views.py](/F:/saiha/alaina/saiha/views.py:644), [saiha/views.py](/F:/saiha/alaina/saiha/views.py:677), [saiha/views.py](/F:/saiha/alaina/saiha/views.py:775), [saiha/views.py](/F:/saiha/alaina/saiha/views.py:962), [saiha/views.py](/F:/saiha/alaina/saiha/views.py:998)
- Problem:
  Many state-changing endpoints are decorated with `@csrf_exempt` even though they operate on authenticated sessions and mutate business data.
- Impact:
  A malicious third-party page can induce logged-in users or corporate admins to perform uploads, credit changes, invoice resends, or request processing.
- Recommended fix:
  Remove `@csrf_exempt` from browser-facing authenticated POST endpoints and rely on Django’s CSRF middleware.

### 5. Demo recharge endpoint is live in normal routing

- Severity: High
- Files: [saiha/urls.py](/F:/saiha/alaina/saiha/urls.py:29), [saiha/views.py](/F:/saiha/alaina/saiha/views.py:610)
- Problem:
  `simulate_corporate_recharge` is exposed as a normal route and directly credits a corporate account without any payment verification.
- Impact:
  Any corporate admin can mint credits outside a payment flow. In many deployments that becomes a revenue, audit, and trust problem rather than a harmless demo shortcut.
- Recommended fix:
  Restrict this endpoint to `DEBUG` only, staff-only admin tooling, or remove it from production routing.

### 6. Corporate admin gate ignores `is_active`

- Severity: High
- Files: [saiha/views.py](/F:/saiha/alaina/saiha/views.py:439), [saiha/views.py](/F:/saiha/alaina/saiha/views.py:1007)
- Problem:
  `corporate_admin_required` checks role only, not whether the corporate profile is active. `api_process_credit_request` does the same.
- Impact:
  A deactivated admin profile could continue accessing admin features if the role remains `ADMIN`.
- Recommended fix:
  Require both `profile.is_active` and `profile.role == ADMIN` in every corporate-admin authorization path.

## Medium

### 7. Standard dataset delete path returns no response

- Severity: Medium
- File: [saiha/views.py](/F:/saiha/alaina/saiha/views.py:230)
- Problem:
  `delete_dataset` only returns an `HttpResponse` for HTMX requests. The non-HTMX success path falls off the end of the view without returning a response.
- Impact:
  Standard browser/API calls will raise a Django error after the dataset is already soft-deleted.
- Recommended fix:
  Return a redirect or JSON success payload for the normal request path.

### 8. Reallocating credits can produce invalid negative user quota limits

- Severity: Medium
- Files: [saiha/views.py](/F:/saiha/alaina/saiha/views.py:679), [saiha/corporate_service.py](/F:/saiha/alaina/saiha/corporate_service.py:104)
- Problem:
  `corporate_reallocate_credits` accepts any float and `reallocate_credits` does not block negative `new_credit_limit` values.
- Impact:
  A bad request can store negative `max_tokens`, which corrupts quota math and downstream billing/usage displays.
- Recommended fix:
  Validate `new_credit_limit >= 0` and preferably enforce upper/lower bounds through a form or serializer.

### 9. Retail top-up response advertises invoices that are never created

- Severity: Medium
- Files: [saiha/views.py](/F:/saiha/alaina/saiha/views.py:412), [saiha/corporate_service.py](/F:/saiha/alaina/saiha/corporate_service.py:430)
- Problem:
  `user_topup` returns `invoice_id` from the latest invoice, but `recharge_user` no longer creates one.
- Impact:
  The API can return a stale or unrelated invoice ID, which is misleading and can send the UI to the wrong billing record.
- Recommended fix:
  Either create a fresh invoice in `recharge_user` or remove `invoice_id` from the retail top-up response.

## Configuration Risks

### 10. Insecure development defaults are too permissive if reused in production

- Severity: Medium
- File: [alaina/settings.py](/F:/saiha/alaina/alaina/settings.py:15)
- Problem:
  `SECRET_KEY` has a hardcoded fallback, `DEBUG` defaults to `True`, and there is no visible production-only hardening for cookies, HTTPS, or host validation beyond a simple env-split.
- Impact:
  Misconfigured deployments can boot in an insecure mode instead of failing closed.
- Recommended fix:
  Fail startup when required production secrets/settings are missing, and add explicit secure-cookie / SSL redirect settings for production.

## Test Coverage Gaps

### 11. No automated coverage for the riskiest flows

- Severity: Medium
- File: [saiha/tests.py](/F:/saiha/alaina/saiha/tests.py:1)
- Problem:
  The Django test module is empty and `manage.py test` reports `0` tests.
- Impact:
  Access-control bugs, quota logic regressions, and async workflow failures can ship undetected.
- Recommended fix:
  Add tests for:
  - session ownership in `api_chat_analysis`
  - websocket authorization
  - invitation acceptance rules
  - credit reallocation bounds
  - dataset deletion responses

## Suggested Fix Order

1. Lock down session ownership in chat analysis and websocket subscriptions.
2. Remove `force_link` from the corporate join flow and switch to token-based invite lookup.
3. Remove `@csrf_exempt` from authenticated mutation endpoints.
4. Disable or gate the simulate-recharge route.
5. Fix response correctness and quota validation issues.
