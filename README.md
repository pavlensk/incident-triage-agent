# AI Incident Triage Assistant

A robust, production-ready AI agent designed to help on-call engineers triage incidents. 
Focuses on architectural cleanliness, strict JSON data contracts, and LLM safety mechanics.

## How to Run (Local Environment)

1. **Copy the environment template and add your API key:**

```bash
   cp .env.example .env
   # Edit .env and insert your OPENAI_API_KEY
```

2. **Install dependencies in a virtual environment:**

```bash
   python -m venv venv
   # Windows: venv\Scripts\activate
   # Linux/Mac: source venv/bin/activate
   pip install -r requirements.txt          # runtime only
   pip install -r requirements-dev.txt      # runtime + test tools (pytest, pytest-cov)
```

3. **Start the FastAPI server:**

```bash
   uvicorn app.main:app --reload
```

4. **Access the UI:**
   Navigate to http://localhost:8000 in your browser.

---

## Testing

### Testing Strategy

The test suite is organized into four layers.  Each layer has a distinct scope
and purpose; together they provide coverage of correctness, reliability, and
output quality without requiring a real OpenAI API key.

| Layer | Location | Purpose |
|-------|----------|---------|
| **Unit** | `tests/unit/` + `tests/test_agent.py` | Isolated components — no I/O, no HTTP |
| **Integration** | `tests/test_api.py` | HTTP layer wiring — TestClient + DI overrides |
| **End-to-End** | `tests/e2e/` | Full scenarios — HTTP → pipeline → response |
| **Evaluations** | `tests/evals/` | Output quality — taxonomy, severity, retrieval |

#### Unit tests (`tests/unit/`, `tests/test_agent.py`)

Test individual classes and functions in isolation.  The LLM is replaced by
`MockLLMClient` — a deterministic stub that returns pre-set strings — so there
are no network calls, no API keys, and no flakiness.

Covered:
- `test_schemas.py` — every Pydantic field constraint at its boundary value
  (min/max length, literal enum, list size).
- `test_exceptions.py` — inheritance chain and catch-at-right-level behaviour.
- `test_agent.py` — `IncidentAnalyzer` pipeline stages, schema-validation retry
  loop, rate-limit exponential backoff, auth-error propagation.
- Retriever keyword extraction and stop-word filtering.
- `PromptBuilder` system prompt and self-correction message content.

#### Integration tests (`tests/test_api.py`)

Test the HTTP layer end-to-end using FastAPI's `TestClient`.
`app.dependency_overrides` replaces the real `IncidentAnalyzer` with a
mock-backed version so tests are still offline and deterministic.

Covered:
- HTTP 200 with valid payload and structured JSON response.
- HTTP 400 for too-short incident text (rejected before LLM call).
- HTTP 422 for missing request body field (Pydantic) and for LLM
  exhausting all validation retries.
- HTTP 503 for transient LLM unavailability and rate-limit errors.
- HTTP 500 for authentication failures (server misconfiguration).
- Structured error response format `{"code": "...", "message": "..."}`.
- Static frontend served at root `/`.

#### End-to-End tests (`tests/e2e/`)

Simulate realistic user interactions with the full application stack.
Only the LLM call is mocked (via `MockLLMClient`); every other component
— routing, retriever, prompt builder, Pydantic validation — is the real
production implementation.

Covered:
- All five canonical incident scenarios from the assignment specification.
- Verify the 400 guard fires before the pipeline is entered.
- Verify the self-correction loop is transparent to the HTTP caller (still 200).

Gold-standard LLM responses live in `tests/e2e/conftest.py`.

#### Evaluation tests (`tests/evals/`)

Evaluate the *quality* of the system's output, not just its mechanical
correctness.  Evals answer: "Does the system produce the right answer for
known inputs?"

| File | What it evaluates |
|------|-------------------|
| `test_taxonomy.py` | Category accuracy for all 6 taxonomy entries |
| `test_severity.py` | Severity rubric adherence (high/medium/low) |
| `test_retrieval.py` | Retriever precision, recall, fallback, keyword extraction |

Gold-standard LLM responses per category live in `tests/evals/conftest.py`.
To add a new eval scenario: add a fixture there and a parametrize row in
the relevant test file — no other changes required.

### Running Tests

```bash
# Run the entire suite
pytest tests/ -v

# Run a single layer
pytest tests/unit/ -v
pytest tests/test_api.py -v
pytest tests/e2e/ -v
pytest tests/evals/ -v

# Run with coverage report (requires pytest-cov)
pytest tests/ --cov=app --cov-report=term-missing --cov-fail-under=80
```

No API key is required to run any test.

### Manual Test Cases & Expected Outputs

Paste these exact multiline inputs into the UI to verify the core logic and structured JSON validation contracts.

***

### Test Case 1: External Provider Issue (Canonical Example)

* **Exact Input:**
Customers complain that card payments often fail, and transactions do not go through.
payment-service logs show many timeouts when calling PayGate, starting from 12:05 UTC.
Other services look normal.

* **Expected Output JSON:**
{
  "category": "External payment provider issue",
  "severity": "high",
  "severity_reason": "Massive card payment failures directly impact core business revenue stream.",
  "affected_users": "All customers attempting to checkout via card payments",
  "summary": "The external provider PayGate is not responding in time, causing mass card payment failures.",
  "hypotheses": [
```json
{
  "title": "Degradation or incident on the PayGate side",
  "reasoning": "Timeouts are observed exclusively when making outbound calls to PayGate, while all internal downstream services remain stable.",
  "next_steps": [
    "Check PayGate public status page and recent provider maintenance notifications.",
    "Compare error rates and latency metrics across alternative payment providers.",
    "Temporarily reroute card payment traffic to a backup payment gateway if available."
  ]
}
```
  ]
}

***

### Test Case 2: Database and Reporting Degradation

* **Exact Input:**
Sharp increase in response time for /payments/create endpoint (up to 5-7 seconds).
PostgreSQL dashboards show high CPU utilization and multiple long-running active queries from reporting-service.
Some clients receive 504 Gateway Timeout from api-gateway.

* **Expected Output JSON:**
{
  "category": "Database resource saturation caused by heavy analytical load",
  "severity": "high",
  "severity_reason": "Core transaction processing is delayed, leading to cascading API timeouts and gateway drops.",
  "affected_users": "Active users attempting to create new transactions and complete checkouts.",
  "summary": "Heavy analytical queries triggered by reporting-service are exhausting PostgreSQL CPU resources, blocking payment-service DB connections.",
  "hypotheses": [
```json
{
  "title": "PostgreSQL connection pool exhaustion and CPU spikes due to reporting queries",
  "reasoning": "Reporting queries are scanning large transactional tables without proper read-replica isolation, locking shared database resources.",
  "next_steps": [
    "Execute a query on pg_stat_activity to identify and terminate long-running analytical processes.",
    "Check connection pool saturation metrics on the payment-service database instance.",
    "Verify if heavy analytical exports can be forcefully restricted during peak hours."
  ]
}
```
  ]
}

***

### Test Case 3: Silent Authentication Failure

* **Exact Input:**
Mobile application users report consistent login failures.
auth-service logs show an explicit spike in 401 Unauthorized responses.
Internal log messages indicate invalid token signatures while other services function normally.

* **Expected Output JSON:**
{
  "category": "Authentication service token verification failure",
  "severity": "high",
  "severity_reason": "Users are entirely unable to log into the platform, completely blocking application access.",
  "affected_users": "All mobile application users attempting to authenticate or refresh sessions.",
  "summary": "The auth-service is rejecting incoming authentication tokens due to verification signature mismatches.",
  "hypotheses": [
```json
{
  "title": "Asymmetric JWT signing key mismatch or out-of-sync vault secrets",
  "reasoning": "Invalid token signatures imply that the keys used to sign the tokens do not match the keys used for verification across distributed service pods.",
  "next_steps": [
    "Check recent secret rotation logs in Vault or the configuration management system.",
    "Verify token verification public key synchronization across all active auth-service pods.",
    "Inspect auth-service pod logs for explicit cryptographic or key loading errors."
  ]
}
```
  ]
}

***

### Test Case 4: Notification SMTP Integration Degradation

* **Exact Input:**
Users report they are not receiving top-up confirmation emails.
Financial balances are credited successfully and billing records are completely correct.
notification-service logs display intermittent connection timeouts to the external SMTP provider.

* **Expected Output JSON:**
{
  "category": "Notification delivery gateway degradation",
  "severity": "low",
  "severity_reason": "Core financial operations are working correctly, impact is limited to delayed non-critical communication.",
  "affected_users": "Customers expecting immediate email alerts for successful balance top-ups.",
  "summary": "The notification-service is experiencing networking connection timeouts with the external upstream SMTP server.",
  "hypotheses": [
```json
{
  "title": "Upstream SMTP provider outage or network routing issues",
  "reasoning": "Intermittent connection timeouts point toward network-level drops or throttling on the external provider side, while internal billing is healthy.",
  "next_steps": [
    "Verify network line connectivity and latency to the remote SMTP endpoint from inside the cluster.",
    "Inspect the internal notification retry queue size and dead-letter queue metrics.",
    "Check external email gateway status dashboards for known provider issues."
  ]
}
```
  ]
}

***

### Test Case 5: Mixed Multi-Layered Degradation (Stress Test)

* **Exact Input:**
System performance dashboard shows payments are slow (averaging 5 seconds per request).
Simultaneously, customer support reports users are not receiving SMS confirmations.
Logs show severe reporting-service load on the primary DB concurrent with network timeout errors to external SMS gateways.

* **Expected Output JSON:**
{
  "category": "Compound multi-service infrastructure degradation",
  "severity": "high",
  "severity_reason": "Simultaneous degradation of payment processing and user notification delivery channels.",
  "affected_users": "Customers executing payments and users expecting multi-factor SMS codes or transaction alerts.",
  "summary": "The platform is experiencing concurrent bottlenecks: internal database CPU saturation and external network connection drops to SMS providers.",
  "hypotheses": [
```json
{
  "title": "Cascading database connection delays combined with independent external SMS gateway timeout",
  "reasoning": "The payment slowness correlates with reporting-service DB load, while the SMS failure is explicitly logged as an external network timeout.",
  "next_steps": [
    "Isolate database metrics to confirm if connection acquisition time is driving payment latency.",
    "Check upstream SMS API endpoint availability and verify external gateway rate limits.",
    "Evaluate the need to provision immediate read-replicas for heavy reporting queries."
  ]
}
```
  ]
}

---

## Architecture

The agent is structured as an explicit multi-stage pipeline aligned with SOLID principles.
Each component has a single, well-defined responsibility and communicates through clean interfaces.

### Module Structure

```text
app/
  main.py           — FastAPI app, lifespan lifecycle, global exception handlers, HTTP routing
  settings.py       — Centralised pydantic-settings configuration (single source of truth)
  schemas.py        — Pydantic data contracts (IncidentAnalysis, Hypothesis)
  context.py        — Static knowledge base: system architecture + past incidents
  agent/
    __init__.py
    analyzer.py     — Orchestrator: coordinates pipeline stages; LLM-level retry with backoff
    retriever.py    — Stage 1+2: input parsing & keyword-based context retrieval
    prompt_builder.py — Stage 3: system prompt assembly (taxonomy, severity rubric, schema injection)
    llm_client.py   — Stage 4: LLMClientProtocol + OpenAI implementation with typed exception mapping
    exceptions.py   — Typed domain exception hierarchy (see Error Handling section)
static/
  index.html        — Simple web UI
tests/
  conftest.py       — Shared fixtures and MockLLMClient stub
  test_agent.py     — Unit tests (pipeline, retry logic, retriever, prompt builder)
  test_api.py       — Integration tests (HTTP layer, all status codes, error response format)
```

### Pipeline Stages

The request flows through four explicit, isolated stages:

```
User Input
    │
    ▼
[1] ContextRetriever.parse_input()   — normalise text, extract keywords
    │
    ▼
[2] ContextRetriever.retrieve()      — keyword-overlap RAG against past incidents
    │
    ▼
[3] PromptBuilder.build_system_prompt()  — inject architecture + context + taxonomy + JSON schema
    │
    ▼
[4] LLMClientProtocol.complete()     — call LLM (with per-request timeout)
        │ RateLimitError?
        └─► exponential backoff retry (up to llm_retry_attempts)
        │ ValidationError?
        └─► self-correction loop (up to max_retries): append error → ask LLM to fix
    │
    ▼
Structured IncidentAnalysis JSON
```

### Key Architectural Decisions

**Dependency Inversion (LLMClientProtocol).** Business logic depends on the `LLMClientProtocol`
abstract interface, not the concrete `openai` SDK. Tests inject `MockLLMClient` — a simple stub
returning pre-set strings — without any monkey-patching. Swapping OpenAI for Anthropic or a local
model requires only a new `LLMClientProtocol` implementation.

**Single Responsibility.** `IncidentAnalyzer` only orchestrates. Parsing lives in `ContextRetriever`,
prompt text lives in `PromptBuilder`, schema lives in `schemas.py`. Each module can be tested
and evolved independently.

**Centralised Configuration.** `Settings` (pydantic-settings) validates all environment variables
at startup. No `os.getenv()` calls are scattered across business-logic modules. LLM timeout and
retry policy are configurable via environment variables (`LLM_TIMEOUT_SECONDS`,
`LLM_RETRY_ATTEMPTS`, `LLM_RETRY_DELAY_SECONDS`).

**Schema Injection.** The exact Pydantic JSON schema is dynamically injected into the system prompt,
enforcing a rigorous data contract on the LLM output.

**Two Independent Retry Mechanisms.** The pipeline defends against two distinct failure modes:
- *Schema-validation retry (self-correction loop)*: if the LLM returns invalid JSON or violates
  field constraints, Pydantic raises a `ValidationError`. The orchestrator appends the error
  details to the conversation and requests a fix — up to `max_retries` times.
- *Transient-error retry with exponential backoff*: if the LLM API returns a rate-limit error,
  `_call_llm()` retries with delay `base * 2^(attempt-1)` before giving up. Authentication
  failures are never retried — they indicate a permanent misconfiguration.

**Typed Domain Exception Hierarchy.** All OpenAI SDK errors are translated at the client boundary
into typed domain exceptions. Upstream code never imports from `openai`:

```
LLMAuthenticationError          — invalid API key (permanent)  → HTTP 500
LLMUnavailableError             — connection / timeout / other → HTTP 503
  └── LLMRateLimitError         — rate limit exceeded          → HTTP 503 (retried first)
LLMInvalidResponseError         — schema validation exhausted  → HTTP 422
```

**Global Exception Handlers.** Three `@app.exception_handler` decorators in `main.py` translate
domain exceptions into HTTP responses with a consistent structured format:

```json
{"code": "llm_unavailable", "message": "..."}
```

This keeps route handlers thin (no try/except for domain errors) and guarantees a uniform
error contract across all endpoints. Authentication errors return a safe, non-leaking message
that does not expose internal API key details to the client.

**FastAPI Lifespan.** All initialisation (logger, LLM client, analyzer) happens inside the
`lifespan` async context manager — not at module import time. This eliminates import-time
side effects and makes the startup/teardown sequence explicit and testable.

---

## Trade-offs & Design Decisions

### Keyword Retrieval vs. Vector Search
The `ContextRetriever` uses keyword overlap (token matching) rather than vector embeddings.
This was a deliberate trade-off: it requires zero infrastructure, has no runtime cost, and is
fully testable offline.  The `InputParser`/`ContextRetriever` split means the keyword backend
can be replaced with pgvector or FAISS by implementing the same `retrieve(parsed_data)` interface
in a new class — no changes to the orchestrator or tests required.

### Free-form `category` vs. Strict `Literal`
The `category` field in `IncidentAnalysis` is a free-form `str` rather than a `Literal` over
`TAXONOMY_CATEGORIES`.  Enforcing a strict enum would trigger unnecessary self-correction
retries for responses that are semantically correct but phrased slightly differently.
Category accuracy is validated separately in `tests/evals/test_taxonomy.py` using
gold-standard fixtures, which gives the same regression coverage without imposing runtime cost.

### JSON Mode vs. Function Calling
The OpenAI `response_format={"type": "json_object"}` mode is used instead of function calling.
JSON mode is simpler to implement, works with all GPT-4 model variants, and the explicit schema
injected into the system prompt provides equivalent structural guarantees.  Function calling
would add stronger type enforcement at the API level but requires a schema serialisation step
and is harder to test offline.

### In-memory Knowledge Base
`PAST_INCIDENTS_LIST` is a static list defined in `context.py`.  For a production deployment,
this would be replaced by a database or vector store queried at runtime.  The current design
keeps the architecture correct (retrieval is isolated behind `ContextRetriever`) while avoiding
operational dependencies for the proof-of-concept.

### No Authentication on the API
The `/api/v1/analyze` endpoint has no authentication layer.  In production, an API gateway or
reverse proxy would enforce auth (API keys, OAuth2) before requests reach FastAPI.  Adding it
directly to the FastAPI app would not change the internal architecture.

---

## Future Improvements

The current implementation deliberately prioritises clean architecture over feature completeness.
The following items remain as natural next steps:

### 1. Advanced Retrieval (RAG)
The current keyword-overlap retriever is a functional placeholder.
A production system would use vector embeddings (e.g., pgvector or FAISS) with semantic scoring
to ensure the LLM receives the most relevant historical context without polluting the context window.

### 2. Richer Input Schema
The current `incident_text` field is a raw string.
Real observability pipelines would benefit from a structured input including
`service`, `timestamp`, `environment`, and `correlation_id` fields for end-to-end tracing.

### 3. Prompt Versioning
As the system scales, prompts should be stored in dedicated template files with version tags,
enabling A/B testing and rollback without code changes.
