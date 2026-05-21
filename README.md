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
   pip install -r requirements.txt
```

3. **Start the FastAPI server:**

```bash
   uvicorn app.main:app --reload
```

4. **Access the UI:**
   Navigate to http://localhost:8000 in your browser.

---

## Testing

The project includes pytest tests covering the LLM retry logic, Pydantic validation,
context retrieval, and prompt assembly — all with a deterministic `MockLLMClient` stub
(no real API calls required).

Run the test suite via:

```bash
pytest tests/ -v
```

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
  main.py           — FastAPI app, lifespan lifecycle, HTTP routing
  settings.py       — Centralised pydantic-settings configuration (single source of truth)
  schemas.py        — Pydantic data contracts (IncidentAnalysis, Hypothesis)
  context.py        — Static knowledge base: system architecture + past incidents
  agent/
    __init__.py
    analyzer.py     — Orchestrator: coordinates all pipeline stages
    retriever.py    — Stage 1+2: input parsing & keyword-based context retrieval
    prompt_builder.py — Stage 3: system prompt assembly (taxonomy, severity rubric, schema injection)
    llm_client.py   — Stage 4: LLMClientProtocol + OpenAI implementation
    exceptions.py   — Typed domain exceptions (LLMUnavailableError, LLMInvalidResponseError)
static/
  index.html        — Simple web UI
tests/
  test_agent.py     — Pytest suite (uses MockLLMClient, no real API calls)
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
[4] LLMClientProtocol.complete()     — call LLM, validate with Pydantic
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
at startup. No `os.getenv()` calls are scattered across business-logic modules.

**Schema Injection.** The exact Pydantic JSON schema is dynamically injected into the system prompt,
enforcing a rigorous data contract on the LLM output.

**Self-Correction Loop.** If the LLM returns invalid JSON or violates `min_length` constraints,
Pydantic raises a `ValidationError`. The orchestrator appends the error details to the conversation
history and requests a correction — up to `MAX_RETRIES` times.

**Typed Domain Exceptions.** `LLMUnavailableError` (network/API failure → HTTP 503) and
`LLMInvalidResponseError` (validation exhausted → HTTP 422) allow `main.py` to return
precise HTTP status codes without catching bare `Exception`.

**FastAPI Lifespan.** All initialisation (logger, LLM client, analyzer) happens inside the
`lifespan` async context manager — not at module import time. This eliminates import-time
side effects and makes the startup/teardown sequence explicit and testable.

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

### 3. LLM Resiliency
Explicit request timeouts and circuit-breaker logic around `OpenAILLMClient` would prevent
cascading failures when the upstream API is slow or rate-limiting.

### 4. Prompt Versioning
As the system scales, prompts should be stored in dedicated template files with version tags,
enabling A/B testing and rollback without code changes.
