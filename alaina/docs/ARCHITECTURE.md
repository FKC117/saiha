# Alaina / Saiha Architecture

## 1. What This Project Is

This repository is a Django-based analytics platform that lets authenticated users:

- upload tabular datasets,
- persist them in a processed storage format,
- ask natural-language analysis questions,
- route those questions through an LLM-backed planning layer,
- execute statistical/visual/data-processing tools asynchronously with Celery,
- stream progress and final interpretations back to the browser over WebSockets,
- export results as PPTX or DOCX,
- manage retail and corporate credit/billing flows.

At a high level, the product is a data-analysis chat application with a fairly large internal tool registry and an added corporate account/billing subsystem.

## 2. Runtime Topology

The project is organized around these main runtime pieces:

### Django web app

- Entry point: [manage.py](/F:/saiha/alaina/manage.py)
- Project config: [alaina/settings.py](/F:/saiha/alaina/alaina/settings.py), [alaina/urls.py](/F:/saiha/alaina/alaina/urls.py), [alaina/asgi.py](/F:/saiha/alaina/alaina/asgi.py)
- Main app: [saiha](/F:/saiha/alaina/saiha)

Responsibilities:

- HTTP views and HTML rendering
- auth/session handling
- dataset upload and dashboard pages
- corporate admin/billing flows
- analysis result fetch/export endpoints

### PostgreSQL

Configured as the primary transactional store in [alaina/settings.py](/F:/saiha/alaina/alaina/settings.py:92).

Stores:

- users and auth-related tables
- datasets and dataset column metadata
- sessions, chat messages, analysis results
- AI audit logs and quota usage
- corporate, invitation, invoice, and credit-request records
- Celery task results through `django_celery_results`

### File storage under `media/`

Managed primarily by [storage_manager_parquet.py](/F:/saiha/alaina/saiha/database_processing_logic/storage_manager_parquet.py).

Stores:

- processed dataset files (`parquet` with CSV fallback)
- dataset metadata JSON previews
- logos and branding assets
- corporate logos

### Redis

Configured through `REDIS_URL` in [alaina/settings.py](/F:/saiha/alaina/alaina/settings.py:180).

Used for:

- Celery broker
- Channels layer for websocket group messaging

### Celery workers

Configured in [alaina/celery.py](/F:/saiha/alaina/alaina/celery.py) and task code under [saiha/celery_tasks](/F:/saiha/alaina/saiha/celery_tasks).

Responsibilities:

- async tool execution
- result lifecycle/status tracking
- websocket progress notifications
- post-analysis interpretation generation

### Channels / WebSockets

- ASGI router: [alaina/asgi.py](/F:/saiha/alaina/alaina/asgi.py)
- websocket routes: [saiha/routing.py](/F:/saiha/alaina/saiha/routing.py)
- consumer: [saiha/consumers.py](/F:/saiha/alaina/saiha/consumers.py)

Responsibilities:

- real-time task state updates
- final AI interpretation delivery to the chat UI

### Gemini integration

- Service wrapper: [saiha/llm_management/gemini_service.py](/F:/saiha/alaina/saiha/llm_management/gemini_service.py)
- Planner: [saiha/agents/analysis_planner.py](/F:/saiha/alaina/saiha/agents/analysis_planner.py)
- Interpreter: [saiha/agents/interpretation_agent.py](/F:/saiha/alaina/saiha/agents/interpretation_agent.py)

Responsibilities:

- intent planning
- JSON tool-intent extraction
- natural-language interpretation of tool outputs
- token/cost audit logging

## 3. Main Code Areas

### Project package: `alaina/`

- framework configuration
- ASGI and Celery bootstrap

### App package: `saiha/`

This is the core domain package.

Key modules:

- [views.py](/F:/saiha/alaina/saiha/views.py): HTTP endpoints
- [models.py](/F:/saiha/alaina/saiha/models.py): core data model
- [corporate_service.py](/F:/saiha/alaina/saiha/corporate_service.py): corporate credits, invitations, invoices
- [session_management/session_manager.py](/F:/saiha/alaina/saiha/session_management/session_manager.py): chat/session lifecycle
- [agents/](/F:/saiha/alaina/saiha/agents): planner/orchestrator/interpreter logic
- [analysis_tools/](/F:/saiha/alaina/saiha/analysis_tools): analysis tool implementations and registry
- [database_processing_logic/](/F:/saiha/alaina/saiha/database_processing_logic): upload processing and dataset storage
- [reporting/](/F:/saiha/alaina/saiha/reporting): DOCX/PPTX narrative export flow
- [celery_tasks/](/F:/saiha/alaina/saiha/celery_tasks): async execution

### Templates and static assets

- templates in [templates/](/F:/saiha/alaina/templates)
- source static assets in [static/](/F:/saiha/alaina/static)
- collected static output in [staticfiles/](/F:/saiha/alaina/staticfiles)

Important browser-side modules:

- [templates/index.html](/F:/saiha/alaina/templates/index.html)
- [static/js/chat.js](/F:/saiha/alaina/static/js/chat.js)
- [static/js/websocket.js](/F:/saiha/alaina/static/js/websocket.js)
- [static/js/upload.js](/F:/saiha/alaina/static/js/upload.js)

## 4. Request and Data Flows

### 4.1 Dataset upload flow

1. User uploads a file from the chat landing page or dataset modal.
2. [views.upload_dataset](/F:/saiha/alaina/saiha/views.py:81) receives the file.
3. [DatasetProcessor](/F:/saiha/alaina/saiha/database_processing_logic/dataset_processor.py) validates, parses, cleans, and normalizes columns.
4. A `Dataset` row is created in PostgreSQL.
5. [DatasetStorageManager](/F:/saiha/alaina/saiha/database_processing_logic/storage_manager_parquet.py) writes the processed file to `media/datasets/<user>/<dataset>/`.
6. `DatasetColumn` records are created from extracted schema metadata.
7. [SessionManager.get_or_create_session](/F:/saiha/alaina/saiha/session_management/session_manager.py:16) creates an analysis session tied to the dataset.
8. Frontend redirects into the chat view for that dataset/session.

### 4.2 Analysis chat flow

1. Browser submits a message to [api_chat_analysis](/F:/saiha/alaina/saiha/views.py:158).
2. View instantiates [AnalysisAgent](/F:/saiha/alaina/saiha/agents/analysis_agent.py).
3. Agent stores the user message in chat history.
4. Agent builds recent conversation context and dataset schema text.
5. [AnalysisPlanner](/F:/saiha/alaina/saiha/agents/analysis_planner.py) either:
   - hard-routes obvious intents,
   - filters the relevant tool subset,
   - or calls Gemini for JSON tool planning.
6. Planned tool params are normalized and corrected against schema.
7. `AnalysisResult` placeholder rows are created with dedup IDs.
8. Celery tasks are dispatched for each planned tool.

### 4.3 Async execution flow

1. [execute_analysis_task](/F:/saiha/alaina/saiha/celery_tasks/analysis_tasks.py) starts.
2. Base task updates the `AnalysisResult` status to `RUNNING`.
3. Task loads the session and dataset.
4. Tool instance is resolved from the whitelist registry.
5. Relevant columns are preselected and loaded from stored dataset files.
6. Tool executes through `validate_and_run`.
7. Result JSON is sanitized before database persistence.
8. `AnalysisResult` status is marked `SUCCESS`.
9. Websocket notifications update the UI through Channels.
10. [InterpretationAgent](/F:/saiha/alaina/saiha/agents/interpretation_agent.py) generates narrative output and persists it as an AI chat message.

### 4.4 Realtime UI flow

1. Browser opens `/ws/notifications/<session_id>/`.
2. [NotificationConsumer](/F:/saiha/alaina/saiha/consumers.py) joins the session group.
3. Celery and interpreter code call `send_ws_notification(...)`.
4. Frontend [websocket.js](/F:/saiha/alaina/static/js/websocket.js) updates the typing indicator or renders the final AI message.

### 4.5 Export flow

1. User hits `/api/export/session/<session>/<format>/`.
2. [ReportBuilder](/F:/saiha/alaina/saiha/reporting/report_builder.py) collects successful results and generates executive-summary content.
3. PPTX or DOCX exporter turns that narrative context into a downloadable report.

### 4.6 Corporate/billing flow

The corporate subsystem is centered on:

- [CorporateService](/F:/saiha/alaina/saiha/corporate_service.py)
- corporate-specific views in [views.py](/F:/saiha/alaina/saiha/views.py)
- `Corporate`, `CorporateProfile`, `CorporateInvitation`, `CreditRequest`, `Invoice`, and `UserQuota` models in [models.py](/F:/saiha/alaina/saiha/models.py)

Responsibilities include:

- org membership and invitations
- seat counts and pooled credits
- quota reallocation to members
- recharge/top-up flows
- invoice generation and email delivery

## 5. Core Data Model

### Dataset domain

- `Dataset`: uploaded dataset metadata plus storage paths and lineage
- `DatasetColumn`: per-column schema profile

### Analysis domain

- `AnalysisSession`: user + dataset conversation context
- `ChatMessage`: persistent conversational history
- `AnalysisResult`: tool execution record, status, payload, interpretation
- `AIAuditLog`: LLM prompt/response/token trail

### Configuration/domain support

- `SiteSettings`, `BusinessInfo`, `AppConfiguration`

### Commercial domain

- `UserQuota`
- `CreditPackage`
- `Corporate`
- `CorporateProfile`
- `CorporateInvitation`
- `CreditRequest`
- `Invoice`

## 6. Design Strengths

- Clear split between HTTP, async execution, and interpretation stages.
- Tool execution is behind a registry whitelist instead of free-form LLM action.
- Result persistence is explicit and auditable.
- Dataset processing and storage are separated from view logic.
- Token usage is tracked in database audit logs.
- Corporate logic is centralized in a service layer rather than duplicated across views.

## 7. Architectural Weaknesses

- Security boundaries are inconsistent between HTML views, JSON endpoints, and websockets.
- Business-critical endpoints are frequently marked `@csrf_exempt`, which weakens the safety guarantees Django would normally provide.
- Analysis ownership checks exist in some places but are missing in the central orchestration entry point.
- The websocket consumer trusts a session UUID alone and does not validate membership/ownership.
- Several flows combine operational/demo behavior with production behavior instead of feature-flagging them.
- Test coverage is effectively absent; the repository currently has zero Django tests.

## 8. Recommended Refactor Priorities

### Priority 1: security boundary cleanup

- enforce object ownership checks at every session/invoice/corporate boundary
- remove `@csrf_exempt` from authenticated mutation endpoints
- authenticate and authorize websocket subscriptions
- disable or feature-flag simulation endpoints outside debug/admin environments

### Priority 2: service and API hardening

- move more mutation logic out of views and into explicit service methods
- add request serializers/forms for POST payload validation
- standardize JSON error responses and status codes
- add transaction boundaries where view logic creates DB rows plus files together

### Priority 3: testability

- add request tests for access control
- add service-layer tests for credit allocation/reallocation
- add async/task tests around result creation and failure transitions
- add integration tests for dataset upload and chat-analysis dispatch

### Priority 4: operability

- add a real deployment configuration split for dev vs production
- centralize environment validation on startup
- document worker, Redis, PostgreSQL, and Gemini prerequisites in a setup guide

## 9. Verification Snapshot

I validated the repository with:

- `python manage.py check` -> passed
- `python manage.py test` -> passed, but there are `0` tests

That means the project is structurally loadable in this environment, but it is not meaningfully protected by automated regression tests yet.
