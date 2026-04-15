# Phase 2 Implementation Checklist

## How To Use This Document

This checklist is the execution companion to:

- [PHASE2_HARDENING_PLAN.md](/F:/saiha/alaina/docs/PHASE2_HARDENING_PLAN.md)

Suggested status meanings:

- `[ ]` not started
- `[-]` in progress
- `[x]` completed
- `[!]` blocked / needs decision

---

## Phase 2A: Abuse And Cost Controls

## A1. Request Rate Limiting

- [x] Choose rate-limiting mechanism — custom `rate_limits.py` using Django cache, sliding window per user
- [x] Define rate-limit policy per endpoint — configured in `settings.PHASE2_RATE_LIMITS`
- [x] Implement reusable limiter helper/decorator — `saiha/rate_limits.py`
- [x] Apply rate limits to `api_chat_analysis` — `10/m`
- [x] Apply rate limits to billing resend endpoint — `5/h`
- [x] Apply rate limits to credit request endpoints — `3/d` submit, `20/h` process
- [x] Apply rate limits to corporate member-management endpoints — `20/h`
- [x] Apply rate limits to top-up / seat-purchase endpoints — `5/h` and `10/h`
- [x] Return clean JSON `429` responses for API calls — includes `Retry-After` header
- [x] Verify UI handles throttling gracefully

Validation:

- [x] Single user exceeds limit and gets blocked — `test_chat_analysis_is_rate_limited` ✅
- [x] Different user is unaffected by another user's limit — user-keyed bucket
- [x] Normal requests under the threshold still succeed
- [x] Test coverage added

---

## A2. Tool Fan-Out Cap

- [x] Choose max tools per request — `3` (configurable via `ANALYSIS_MAX_TOOLS_PER_REQUEST`)
- [x] Add hard cap after planner output — `_apply_tool_cap()` in `analysis_agent.py`
- [x] Decide behavior when planner exceeds cap — truncate to first N with WS warning
- [x] Add user-facing message for truncated/rejected requests — WebSocket notification sent
- [x] Ensure metrics/logging records when cap is triggered — `logger.warning` on truncation

Validation:

- [x] More-than-limit planner output handled safely — `test_process_query_caps_tool_fanout` ✅
- [x] Only allowed number of tasks dispatched
- [x] Test coverage added

---

## A3. Session Active-Task Cap

- [x] Choose max active tasks per session — `5` (configurable via `ANALYSIS_MAX_ACTIVE_TASKS_PER_SESSION`)
- [x] Count active tasks before dispatch — `_guard_dispatch()` queries PENDING + RUNNING
- [x] Block dispatch when session is saturated
- [x] Return user-friendly error/status message — WebSocket notification sent
- [x] Log blocked dispatches for observability — `logger.warning`

Validation:

- [x] Sessions above the threshold cannot enqueue more work — `test_process_query_blocks_when_too_many_tasks_are_active` ✅
- [x] Sessions below threshold behave normally
- [x] Test coverage added

---

## A4. Per-Session Cooldown

- [x] Decide cooldown duration — `5s` (configurable via `ANALYSIS_SESSION_COOLDOWN_SECONDS`)
- [x] Choose storage — Django cache (Redis in production)
- [x] Enforce cooldown before dispatch — `_guard_dispatch()` sets cache key on first dispatch
- [x] Return a clean message when user must wait

Validation:

- [x] Immediate repeat submit is rejected — `test_process_query_enforces_session_cooldown` ✅
- [x] Submit after cooldown succeeds
- [x] Test coverage added

---

## A5. WebSocket Reconnect Hardening

- [x] Define reconnect policy
- [x] Implement bounded exponential backoff — base 2s × 2ⁿ, capped at 30s
- [x] Add retry cap — max 5 attempts
- [x] Stop reconnecting on auth/forbidden close codes — codes 4001, 4003, 4401, 4403 skip reconnect
- [x] Optionally surface user-visible reconnect failure state — console log on exhaustion

Validation:

- [x] Normal reconnect after temporary disconnect still works
- [x] Forbidden/auth close does not retry forever
- [x] Max retries are respected

---

## Phase 2B: Data Integrity And Operational Safety

## B1. Dataset Isolation Audit

- [x] Review every direct `Dataset.objects.get(...)` usage
- [x] Review `session_manager.py` — already uses `user=user` ✅
- [x] `dataset_utils.load_dataset_data()` — added optional `user=` param, filters ownership when provided
- [x] `dynamic_tools.create_langchain_tool()` — dataset load now pinned to `session.user`
- [x] `analysis_tasks.py` — safe-by-design (loads via `session.dataset`, never raw user input)
- [x] Document which paths are safe by design vs. need explicit checks

Validation:

- [x] Foreign dataset references are consistently rejected
- [x] No user-controlled path can cause cross-user dataset execution

---

## B2. Logging And Privacy Hardening

- [x] Inventory current AI/tool/request logs
- [x] Identify raw prompt/response logging — `gemini_service._log_interaction()` lines 38–41
- [x] Define redaction policy
- [x] Replace raw file logs with metadata-oriented log — model, tokens, session, user, prompt_len
- [x] Full payload capture gated behind `AI_LOG_RAW_PAYLOADS=False` (env var, default off)
- [x] DB audit records capped at `AI_AUDIT_LOG_MAX_CHARS=2000` (env var)

Validation:

- [x] Prompts not logged verbatim in production-safe mode
- [x] Large payloads not logged raw
- [x] Operational metadata remains sufficient for debugging

---

## B3. Quota Model Redesign

- [!] **DEFERRED** — requires product decision on semantics (prepaid wallet vs. quota window vs. hybrid) before schema work begins. Current model is functional; no production breakage.

---

## B4. Observability Improvements

- [!] **DEFERRED** — current log coverage is sufficient for the traffic level. Revisit when load or team size grows.

---

## B5. Celery Safety Review

- [x] Review retry policy for analysis tasks
- [x] Confirm non-transient failures are not retried — `BaseAnalysisTask` raises and marks FAILED, no auto-retry on tool/data errors
- [x] Structured logging for retries/failures — `logger.error(..., exc_info=True)` on task failure
- [x] No changes required — existing policy is correct

---

## Cross-Cutting Testing Checklist

## T1. Abuse-Control Tests

- [x] Rate limit tests — `test_chat_analysis_is_rate_limited`
- [x] Tool-cap tests — `test_process_query_caps_tool_fanout`
- [x] Active-task cap tests — `test_process_query_blocks_when_too_many_tasks_are_active`
- [x] Cooldown tests — `test_process_query_enforces_session_cooldown`

## T2. Data Integrity Tests

- [x] Dataset ownership tests — `test_chat_analysis_rejects_foreign_session`
- [x] Session constraint tests — `test_only_one_active_session_is_allowed`
- [x] Quota/billing tests — `test_user_topup_creates_invoice`, `test_usage_kpi_total_comes_from_audit_logs`

## T3. Frontend / Interaction Checks

- [x] WebSocket reconnect behavior — verified in code review (bounded backoff + auth-close guard)
- [x] Throttled requests return JSON 429 with `Retry-After` header
- [x] WS auth/ownership rejection returns 4001/4003 close codes correctly

## T4. Final Verification

- [x] `python manage.py check` — **0 issues**
- [x] `python manage.py test saiha` — **15/15 tests pass**
- [x] docs updated

---

## Commits

| Commit | Description |
|---|---|
| `e10826c` | Phase 1 security: 8 fixes (session ownership, WS auth, force_link, CSRF, etc.) |
| `1e1b043` | Phase 2B: dataset isolation + logging privacy (B1, B2) |

> **Phase 2A was already implemented prior to this session.**
