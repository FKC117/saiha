# Phase 2 Implementation Checklist

## How To Use This Document

This checklist is the execution companion to:

- [PHASE2_HARDENING_PLAN.md](/F:/saiha/alaina/docs/PHASE2_HARDENING_PLAN.md)

Use it to track implementation progress during Phase 2.

Suggested status meanings:

- `[ ]` not started
- `[-]` in progress
- `[x]` completed
- `[!]` blocked / needs decision

## Phase 2A: Abuse And Cost Controls

## A1. Request Rate Limiting

Goal:

- protect the highest-risk authenticated POST endpoints from spam, cost abuse, and accidental flooding

Tasks:

- [ ] Choose rate-limiting mechanism
- [ ] Define rate-limit policy per endpoint
- [ ] Implement reusable limiter helper/decorator
- [ ] Apply rate limits to `api_chat_analysis`
- [ ] Apply rate limits to billing resend endpoint
- [ ] Apply rate limits to credit request endpoints
- [ ] Apply rate limits to corporate member-management endpoints
- [ ] Apply rate limits to top-up / seat-purchase endpoints
- [ ] Return clean JSON `429` responses for API calls
- [ ] Verify UI handles throttling gracefully

Likely files:

- [saiha/views.py](/F:/saiha/alaina/saiha/views.py)
- possible new module such as `saiha/rate_limits.py`
- [alaina/settings.py](/F:/saiha/alaina/alaina/settings.py)

Validation:

- [ ] Single user exceeds limit and gets blocked
- [ ] Different user is unaffected by another user’s limit
- [ ] Normal requests under the threshold still succeed
- [ ] Test coverage added

## A2. Tool Fan-Out Cap

Goal:

- prevent one request from dispatching too many expensive tools

Tasks:

- [ ] Choose max tools per request
- [ ] Add hard cap after planner output
- [ ] Decide behavior when planner exceeds cap
- [ ] Add user-facing message for truncated/rejected requests
- [ ] Ensure metrics/logging records when cap is triggered

Likely files:

- [saiha/agents/analysis_agent.py](/F:/saiha/alaina/saiha/agents/analysis_agent.py)
- [saiha/agents/analysis_planner.py](/F:/saiha/alaina/saiha/agents/analysis_planner.py)

Validation:

- [ ] More-than-limit planner output is handled safely
- [ ] Only allowed number of tasks is dispatched
- [ ] Test coverage added

## A3. Session Active-Task Cap

Goal:

- stop sessions from accumulating too many `PENDING` / `RUNNING` jobs

Tasks:

- [ ] Choose max active tasks per session
- [ ] Count active tasks before dispatch
- [ ] Block dispatch when session is saturated
- [ ] Return user-friendly error/status message
- [ ] Log blocked dispatches for observability

Likely files:

- [saiha/agents/analysis_agent.py](/F:/saiha/alaina/saiha/agents/analysis_agent.py)
- [saiha/models.py](/F:/saiha/alaina/saiha/models.py) if helper methods are added

Validation:

- [ ] Sessions above the threshold cannot enqueue more work
- [ ] Sessions below threshold behave normally
- [ ] Test coverage added

## A4. Per-Session Cooldown

Goal:

- prevent rapid repeated submissions from double-clicking or chat spam

Tasks:

- [ ] Decide cooldown duration
- [ ] Choose storage location for last-dispatch timestamp
- [ ] Enforce cooldown before dispatch
- [ ] Return a clean message when user must wait

Likely files:

- [saiha/agents/analysis_agent.py](/F:/saiha/alaina/saiha/agents/analysis_agent.py)
- [saiha/models.py](/F:/saiha/alaina/saiha/models.py) if a new session field is needed

Validation:

- [ ] Immediate repeat submit is rejected
- [ ] Submit after cooldown succeeds
- [ ] Test coverage added

## A5. WebSocket Reconnect Hardening

Goal:

- prevent infinite reconnect loops and noisy retry storms

Tasks:

- [ ] Define reconnect policy
- [ ] Implement bounded exponential backoff
- [ ] Add retry cap
- [ ] Stop reconnecting on auth/forbidden close codes
- [ ] Optionally surface user-visible reconnect failure state

Likely files:

- [static/js/websocket.js](/F:/saiha/alaina/static/js/websocket.js)

Validation:

- [ ] Normal reconnect after temporary disconnect still works
- [ ] Forbidden/auth close does not retry forever
- [ ] Max retries are respected

## Phase 2B: Data Integrity And Operational Safety

## B1. Dataset Isolation Audit

Goal:

- ensure ownership is consistently enforced for all dataset and session paths

Tasks:

- [ ] Review every direct `Dataset.objects.get(...)` usage
- [ ] Review every session-to-dataset transition path
- [ ] Review helper loaders that accept raw dataset IDs
- [ ] Document which paths are safe by design vs. need explicit checks
- [ ] Add/strengthen ownership guards where needed

Likely files:

- [saiha/views.py](/F:/saiha/alaina/saiha/views.py)
- [saiha/session_management/session_manager.py](/F:/saiha/alaina/saiha/session_management/session_manager.py)
- [saiha/database_processing_logic/dataset_utils.py](/F:/saiha/alaina/saiha/database_processing_logic/dataset_utils.py)
- [saiha/celery_tasks/analysis_tasks.py](/F:/saiha/alaina/saiha/celery_tasks/analysis_tasks.py)

Validation:

- [ ] Foreign dataset references are consistently rejected
- [ ] No user-controlled path can cause cross-user dataset execution
- [ ] Test coverage added

## B2. Logging And Privacy Hardening

Goal:

- reduce sensitive payload exposure in logs while preserving debuggability

Tasks:

- [ ] Inventory current AI/tool/request logs
- [ ] Identify raw prompt/response logging
- [ ] Identify raw data payload logging
- [ ] Define redaction policy
- [ ] Replace raw logs with metadata-oriented logs
- [ ] Add production-safe defaults

Likely files:

- [saiha/llm_management/gemini_service.py](/F:/saiha/alaina/saiha/llm_management/gemini_service.py)
- [saiha/celery_tasks/analysis_tasks.py](/F:/saiha/alaina/saiha/celery_tasks/analysis_tasks.py)
- [alaina/settings.py](/F:/saiha/alaina/alaina/settings.py)

Validation:

- [ ] Prompts are no longer logged verbatim in production-safe mode
- [ ] Large payloads are not logged raw
- [ ] Operational metadata remains sufficient for debugging

## B3. Quota Model Redesign

Goal:

- separate lifetime usage from spendable balance and stop overloading one field for both meanings

Tasks:

- [ ] Choose canonical quota/accounting model
- [ ] Write field-level semantics
- [ ] Design schema changes
- [ ] Add migration(s)
- [ ] Update usage accumulation logic
- [ ] Update recharge logic
- [ ] Update corporate allocation logic if needed
- [ ] Update dashboard/reporting logic
- [ ] Backfill or safely initialize existing rows

Likely files:

- [saiha/models.py](/F:/saiha/alaina/saiha/models.py)
- [saiha/corporate_service.py](/F:/saiha/alaina/saiha/corporate_service.py)
- [saiha/llm_management/gemini_service.py](/F:/saiha/alaina/saiha/llm_management/gemini_service.py)
- [saiha/views.py](/F:/saiha/alaina/saiha/views.py)
- [static/js/usage_dashboard.js](/F:/saiha/alaina/static/js/usage_dashboard.js)

Validation:

- [ ] Recharge preserves lifetime usage
- [ ] Available balance changes correctly after usage
- [ ] Dashboard values match the new model
- [ ] Test coverage added

## B4. Observability Improvements

Goal:

- make it easier to answer “what is slow / failing / expensive?”

Tasks:

- [ ] Define key structured events
- [ ] Add timing around planning
- [ ] Add timing around tool execution
- [ ] Add timing around interpretation
- [ ] Add logs/metrics for blocked rate-limited requests
- [ ] Add logs/metrics for fan-out cap triggers
- [ ] Add logs/metrics for active-task cap triggers

Likely files:

- [saiha/agents/analysis_agent.py](/F:/saiha/alaina/saiha/agents/analysis_agent.py)
- [saiha/celery_tasks/analysis_tasks.py](/F:/saiha/alaina/saiha/celery_tasks/analysis_tasks.py)
- [alaina/settings.py](/F:/saiha/alaina/alaina/settings.py)

Validation:

- [ ] Timing data is visible in logs/metrics
- [ ] Failure reasons are easier to distinguish
- [ ] No sensitive payload regression introduced by new instrumentation

## B5. Celery Safety Review

Goal:

- prevent expensive retry storms and make task-failure behavior predictable

Tasks:

- [ ] Review retry policy for analysis tasks
- [ ] Confirm non-transient tool/data failures are not retried
- [ ] Add structured logging for retries/failures
- [ ] Consider failure thresholds or alerting hooks

Likely files:

- [saiha/celery_tasks/base.py](/F:/saiha/alaina/saiha/celery_tasks/base.py)
- [saiha/celery_tasks/analysis_tasks.py](/F:/saiha/alaina/saiha/celery_tasks/analysis_tasks.py)

Validation:

- [ ] Deterministic failures do not loop
- [ ] Retry behavior is understandable from logs

## Cross-Cutting Testing Checklist

## T1. Abuse-Control Tests

- [ ] Rate limit tests
- [ ] Tool-cap tests
- [ ] Active-task cap tests
- [ ] Cooldown tests

## T2. Data Integrity Tests

- [ ] Dataset ownership tests
- [ ] Quota-model tests
- [ ] Recharge/use/recharge scenario tests

## T3. Frontend / Interaction Checks

- [ ] WebSocket reconnect behavior manually verified
- [ ] Throttled requests show understandable user feedback
- [ ] Blocked dispatches do not leave the UI in a stuck “loading” state

## T4. Final Verification

- [ ] `python manage.py makemigrations --check`
- [ ] `python manage.py check`
- [ ] `python manage.py test`
- [ ] docs updated to reflect implemented protections

## Decisions Needed Before Or During Phase 2

- [!] Choose rate-limiting library or implementation style
- [!] Decide maximum tools per request
- [!] Decide maximum active tasks per session
- [!] Decide cooldown duration
- [!] Decide whether to keep raw AI prompt/response capture as an optional debug-only feature
- [!] Finalize quota/accounting semantics before schema redesign

## Suggested Milestones

## Milestone 1

Abuse protection baseline:

- rate limiting
- tool cap
- active-task cap
- cooldown

## Milestone 2

Client/runtime resilience:

- websocket reconnect hardening
- better blocked-request messaging

## Milestone 3

Data and privacy safety:

- dataset isolation audit
- logging/privacy hardening

## Milestone 4

Accounting correctness:

- quota redesign
- dashboard alignment

## Milestone 5

Operational readiness:

- observability improvements
- Celery safety review
- final test expansion
