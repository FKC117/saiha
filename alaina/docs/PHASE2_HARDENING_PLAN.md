# Phase 2 Hardening Plan

## Purpose

This document captures the next implementation phase after the initial security remediation pass.

Phase 1 addressed the most urgent correctness and access-control issues:

- session ownership enforcement,
- websocket authorization,
- invite-flow hardening,
- CSRF restoration,
- demo endpoint restriction,
- selected business-logic fixes,
- regression test coverage for the repaired paths.

Phase 2 focuses on abuse resistance, cost containment, data integrity, and production-readiness improvements.

The goal is to move the system from:

- secure for normal authenticated use

to:

- resilient under misuse,
- safer for sensitive data,
- more predictable under load,
- easier to operate in production.

## Current State Summary

The system is now materially safer than before, but it still has important non-trivial gaps:

- authenticated users can still flood expensive analysis endpoints,
- one prompt can still fan out into too many tool executions,
- task concurrency is not tightly controlled,
- websocket reconnect behavior is still unbounded in non-auth failure cases,
- logs may capture more user/data detail than is appropriate,
- quota/accounting semantics are still too tightly coupled,
- observability is still mostly log-centric rather than metrics-oriented.

These are not “broken app” issues in the same way as the original auth bugs, but they are exactly the kinds of issues that become expensive in production.

## Phase 2 Objectives

### Objective 1: Prevent abuse and cost spikes

Protect:

- `/api/chat-analysis/`
- corporate/admin mutation endpoints
- async execution fan-out
- websocket reconnect churn

Success means:

- a single user cannot rapidly queue excessive analysis work,
- one request cannot trigger an unreasonable number of tool executions,
- one session cannot accumulate too many active async tasks,
- client reconnect behavior does not create unnecessary load.

### Objective 2: Improve data integrity and accounting correctness

Protect:

- quota semantics,
- usage reporting,
- balance tracking,
- dataset ownership assumptions.

Success means:

- “lifetime usage” and “available balance” are modeled separately,
- recharges no longer distort reporting semantics,
- every dataset access path is explicitly ownership-safe.

### Objective 3: Reduce operational and privacy risk

Protect:

- sensitive logs,
- Celery stability,
- production deploy safety,
- future debugging workflows.

Success means:

- logs do not unnecessarily capture raw sensitive content,
- failing tools do not create uncontrolled retry pressure,
- the system exposes enough runtime information to debug failures responsibly.

## Workstreams

## Workstream A: Request Rate Limiting

### Why

Even with ownership and auth fixed, authenticated abuse is still possible.

Examples:

- repeated chat-analysis POSTs,
- repeated corporate credit operations,
- repeated invoice resend requests,
- repeated credit-request processing.

Without throttling, this can become:

- cost spikes from LLM calls,
- Celery backlog growth,
- Redis pressure,
- degraded user experience for everyone else.

### Proposed changes

Implement rate limiting at the Django request layer.

Initial target endpoints:

- `saiha.views.api_chat_analysis`
- `saiha.views.user_topup`
- `saiha.views.corporate_topup`
- `saiha.views.corporate_purchase_seats`
- `saiha.views.api_resend_invoice`
- `saiha.views.api_submit_credit_request`
- `saiha.views.api_process_credit_request`
- `saiha.views.corporate_add_member`
- `saiha.views.corporate_resend_invite`
- `saiha.views.corporate_remove_member`
- `saiha.views.corporate_reallocate_credits`

### Design approach

Preferred implementation:

- use a request-level throttling library or decorator pattern,
- key primarily by authenticated user ID,
- fall back to IP-based throttling for unauthenticated sensitive entry points if needed later.

Suggested initial thresholds:

- chat-analysis: `10/minute` per user
- invoice resend: `5/hour` per user
- corporate member management actions: `20/hour` per admin
- credit request submission: `3/day` per user
- top-up / seat purchase actions: low-volume, stricter limits

### Expected file touch points

- [saiha/views.py](/F:/saiha/alaina/saiha/views.py)
- possibly a new utility module such as `saiha/rate_limits.py`
- settings if a cache backend or shared throttle storage is needed

### Verification

- request tests for 429 responses after threshold breach
- ensure normal usage still works under threshold
- verify no endpoint silently fails with HTML instead of JSON

## Workstream B: Tool Fan-Out and Session Concurrency Limits

### Why

The planner/executor path can still be expensive even for a single authorized request.

The current shape is:

1. user submits one chat message
2. planner returns a list of intents
3. agent loops over intents
4. each intent becomes a Celery task

This means one prompt can still generate too much compute.

### Proposed protections

#### B1. Maximum tools per request

Add a hard cap in the analysis agent after planning.

Suggested cap:

- default max: `3` tools per request

If the planner returns more than the cap:

- either truncate safely,
- or reject with a user-visible explanation,
- or prioritize tools by type/order if deterministic ranking can be introduced.

#### B2. Maximum active tasks per session

Before dispatching new tasks, count currently active `AnalysisResult` rows for the session:

- `PENDING`
- `RUNNING`

If active tasks exceed a threshold:

- reject new work temporarily,
- ask the user to wait until running analyses finish.

Suggested cap:

- `5` active tasks per session initially

#### B3. Short per-session cooldown

Prevent rapid repeat submissions from the same session within a very small window.

Suggested cooldown:

- 3 to 5 seconds between dispatch attempts

This is not a substitute for rate limiting.
It is a second layer specifically for chat spam and double-click behavior.

### Expected file touch points

- [saiha/agents/analysis_agent.py](/F:/saiha/alaina/saiha/agents/analysis_agent.py)
- [saiha/models.py](/F:/saiha/alaina/saiha/models.py) if session metadata needs new fields
- possibly a new helper module for dispatch guards

### Verification

- test planner result truncation or rejection behavior
- test that sessions with too many active tasks cannot enqueue more
- test cooldown behavior across repeated POSTs

## Workstream C: Dataset Isolation Audit

### Why

The main session-level access path is fixed, but data ownership must be true everywhere, not just in the main HTTP entry point.

Any helper path that loads datasets indirectly can become a future bypass if ownership is assumed rather than checked.

### Scope

Audit:

- dataset selection flows,
- session creation,
- dataset detail views,
- helper loaders,
- tool execution paths,
- export paths,
- any utility that resolves `Dataset` from a raw ID.

### Specific files to review carefully

- [saiha/views.py](/F:/saiha/alaina/saiha/views.py)
- [saiha/session_management/session_manager.py](/F:/saiha/alaina/saiha/session_management/session_manager.py)
- [saiha/database_processing_logic/dataset_utils.py](/F:/saiha/alaina/saiha/database_processing_logic/dataset_utils.py)
- [saiha/celery_tasks/analysis_tasks.py](/F:/saiha/alaina/saiha/celery_tasks/analysis_tasks.py)
- any analysis-tool helper that loads dataset state outside the standard session flow

### Target outcome

Make ownership rules explicit:

- HTTP layer validates ownership
- session layer validates ownership
- helper loaders are only reachable through safe contexts
- background execution never runs against a cross-user dataset reference

### Verification

- tests for foreign dataset/session references failing consistently
- code review ensuring there are no `Dataset.objects.get(id=...)` paths exposed through user-controlled entry points without prior ownership checks

## Workstream D: WebSocket Reconnect Hardening

### Why

The websocket auth issue is fixed, but client reconnect behavior is still simplistic.

Current concerns:

- reconnects can continue forever on infrastructure problems,
- reconnect cadence is fixed rather than adaptive,
- there is no upper bound,
- browser churn can become noisy under server outages.

### Proposed changes

Update the reconnect strategy in the frontend websocket client.

Desired behavior:

- do not reconnect on auth/forbidden closes
- use bounded exponential backoff
- stop retrying after a capped number of attempts
- optionally surface a user-visible “reconnect failed” state

### Suggested policy

- base delay: 2 seconds
- exponential multiplier: x2
- cap delay at 30 seconds
- max attempts: 5 to 8

### Expected file touch points

- [static/js/websocket.js](/F:/saiha/alaina/static/js/websocket.js)

### Verification

- manual browser-side testing
- unit-style JS review or lightweight integration tests if available

## Workstream E: Logging and Privacy Hardening

### Why

The platform processes datasets that may contain:

- emails,
- financial values,
- corporate metrics,
- potentially regulated or private data.

Current audit logging is useful, but it risks over-logging raw prompts/responses and potentially raw data-adjacent content.

### Goals

- keep logs operationally useful,
- reduce sensitive payload exposure,
- avoid logging raw data unless explicitly necessary,
- prefer metadata over content.

### Areas to review

- [saiha/llm_management/gemini_service.py](/F:/saiha/alaina/saiha/llm_management/gemini_service.py)
- tool execution logs
- exception logging around analysis payloads
- email/billing logs
- websocket error logs

### Proposed changes

#### E1. Prompt logging minimization

Replace raw prompt logging with:

- prompt length,
- session ID,
- user ID,
- model ID,
- token counts,
- prompt type/category if inferable

Keep raw prompt capture optional and disabled by default in production.

#### E2. Response logging minimization

Avoid logging full LLM response bodies in production logs.

Use:

- truncated summaries,
- status markers,
- token counts,
- result IDs.

#### E3. Data payload redaction

If values are logged:

- redact obvious email-like strings,
- avoid logging row-level data,
- never log large structured dataset payloads by default.

### Verification

- grep-based review for raw payload logging calls
- test/inspect produced logs during common flows

## Workstream F: Quota and Accounting Model Redesign

### Why

This is the most important data-model issue left.

Right now the model still mixes:

- usage accounting,
- available token balance,
- recharge lifecycle behavior.

That causes semantic confusion:

- what is “used” versus “available”,
- what should reset on recharge,
- what should accumulate forever,
- what the dashboard should report.

### Target model

Separate these concerns explicitly.

Suggested conceptual fields:

- `lifetime_tokens_used`
- `current_token_balance`
- `current_plan_allocation` or equivalent if needed
- `expired_token_balance`
- `expiry_date`

Alternative acceptable shape:

- keep `max_tokens` only if it truly means current active allocation ceiling,
- add a distinct lifetime usage field,
- stop deriving business reporting from a field that resets.

### Required decisions

Before implementation, choose which model semantics the product actually wants:

1. prepaid wallet model
2. quota window model
3. hybrid “allocation + usage history” model

The current code behaves like a hybrid but without clear field separation.

### Implementation tasks

#### F1. Define canonical semantics

Document what each quota field means.

#### F2. Add new fields via migration

Likely new fields:

- `lifetime_tokens_used`
- `available_tokens` or equivalent

#### F3. Update usage updates

When LLM calls occur:

- increment lifetime usage
- decrement available balance if applicable

#### F4. Update recharge flows

Recharge should:

- increase balance,
- not erase lifetime usage.

#### F5. Update dashboards

Dashboard cards should map to:

- today usage
- lifetime usage
- current balance
- rescue/expired balance

### Expected file touch points

- [saiha/models.py](/F:/saiha/alaina/saiha/models.py)
- [saiha/corporate_service.py](/F:/saiha/alaina/saiha/corporate_service.py)
- [saiha/llm_management/gemini_service.py](/F:/saiha/alaina/saiha/llm_management/gemini_service.py)
- [saiha/views.py](/F:/saiha/alaina/saiha/views.py)
- [static/js/usage_dashboard.js](/F:/saiha/alaina/static/js/usage_dashboard.js)

### Verification

- migration tests
- recharge/use/recharge scenarios
- corporate member allocation scenarios
- reporting correctness tests

## Workstream G: Observability and Task Safety

### Why

The system has logs, but not enough structured runtime visibility.

When the app is under load, operators will need to answer:

- which tools are slow?
- which sessions are generating the most work?
- which tasks are failing repeatedly?
- where are token costs coming from?

### Proposed changes

#### G1. Execution timing metrics

Capture:

- request start/end timing for chat-analysis
- planning time
- queue time
- tool execution duration
- interpretation duration

#### G2. Structured event logs

Standardize event records for:

- dispatch accepted
- dispatch rejected
- rate limited
- concurrency blocked
- tool started
- tool succeeded
- tool failed

#### G3. Celery safety review

Review retry policies for expensive tasks.

Current state already has selective retry behavior, but Phase 2 should confirm:

- no expensive non-transient failures are retried,
- failure storms are visible,
- repeated tool failure patterns can be detected quickly.

### Expected file touch points

- [saiha/celery_tasks/base.py](/F:/saiha/alaina/saiha/celery_tasks/base.py)
- [saiha/celery_tasks/analysis_tasks.py](/F:/saiha/alaina/saiha/celery_tasks/analysis_tasks.py)
- logging configuration in [alaina/settings.py](/F:/saiha/alaina/alaina/settings.py)

### Verification

- simulate task success/failure
- inspect emitted logs/metrics
- verify no excessive retries for deterministic failures

## Workstream H: Test Expansion

### Why

The newly added tests cover important repaired behaviors, but they do not yet cover abuse resistance or accounting redesign.

### New test categories

#### H1. Rate limiting tests

- threshold reached returns 429
- counters are user-specific

#### H2. Tool cap tests

- planner returns > max intents and agent handles it safely

#### H3. Active-task limit tests

- session with too many `PENDING/RUNNING` tasks is blocked

#### H4. Quota model tests

- lifetime usage is preserved
- recharge increases available balance without resetting lifetime totals

#### H5. Logging/privacy tests

- ensure sensitive raw payloads are not written when redaction mode is enabled

#### H6. Websocket reconnect logic

- mostly client-side/manual unless a JS test harness is introduced

### Expected file touch points

- [saiha/tests.py](/F:/saiha/alaina/saiha/tests.py)
- possibly split into multiple test modules if test volume grows

## Proposed Implementation Order

### Step 1

Request rate limiting

Reason:

- immediate protection against authenticated abuse
- relatively low schema risk
- high operational payoff

### Step 2

Tool-count cap and session active-task cap

Reason:

- contains compute explosion at the orchestration layer

### Step 3

Websocket reconnect hardening

Reason:

- low complexity
- reduces noisy load during outages

### Step 4

Dataset isolation audit and cleanup

Reason:

- ensures no secondary access paths were missed

### Step 5

Logging/privacy hardening

Reason:

- important before handling more sensitive or larger customer datasets

### Step 6

Quota model redesign

Reason:

- highest conceptual complexity
- more invasive schema and reporting changes
- best done after the simpler abuse controls are already in place

### Step 7

Observability improvements and expanded tests

Reason:

- supports long-term maintenance of all previous workstreams

## Deliverables

At the end of Phase 2, the repository should contain:

- request throttling on high-risk endpoints,
- hard limits on tool fan-out,
- hard limits on active session task concurrency,
- bounded websocket reconnect behavior,
- cleaned-up logging/redaction rules,
- explicit quota/accounting semantics with schema support,
- expanded regression tests,
- updated architecture and operational documentation.

## Definition of Done

Phase 2 is complete when all of the following are true:

- abusive request bursts are throttled,
- a single prompt cannot trigger excessive background work,
- sessions cannot accumulate unbounded active analysis tasks,
- websocket reconnects are bounded,
- ownership checks are consistent across dataset/session paths,
- logs are materially safer for sensitive data,
- usage/balance accounting semantics are explicit and correct,
- tests cover the new protections,
- `manage.py check` and tests pass cleanly.

## Recommended Execution Strategy

Implement this phase in two sub-phases:

### Phase 2A

Abuse and cost controls:

- rate limiting
- tool caps
- active-task caps
- websocket reconnect hardening

### Phase 2B

Data/operational correctness:

- quota redesign
- logging/privacy hardening
- observability improvements
- expanded tests and docs

This split keeps the fastest, highest-value controls moving first while reserving the more invasive accounting redesign for a focused pass.
