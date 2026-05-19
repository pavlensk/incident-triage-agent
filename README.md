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

The project includes pytest tests covering the LLM retry logic and Pydantic validation (with mocked OpenAI responses).
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

## High-Level Architecture & Resilience

* **Schema Injection:** The exact Pydantic JSON schema is dynamically injected into the system prompt, enforcing a rigorous data contract on the LLM.
* **Self-Correction (Recovery Loop):** If the LLM generates invalid JSON (or violates min_length constraints), Pydantic raises an error. The orchestrator intercepts this, feeds the specific error back to the LLM, and forces a correction.
* **UI Safety:** All LLM outputs are aggressively escaped on the frontend before DOM insertion to prevent XSS.
* **Configuration:** Core LLM behavior (temperature, model_name, max_retries) is externalized to .env for safe and fast tuning.

## Future Evolution (What I would do with more time)

While the current MVP prioritizes KISS and rapid delivery, scaling this into a Tier-1 production service requires addressing several architectural debts to fully align with SOLID principles.

### 1. Architectural Decoupling (SRP & Dependency Inversion)
* **Deconstruct the "God-Class":** The current IncidentAgent handles parsing, retrieval, prompt assembly, validation, and LLM communication. This violates the Single Responsibility Principle. I would split this into distinct domain components: IncidentParser, ContextRetriever, PromptBuilder, and an orchestrating IncidentAnalyzer.
* **Abstract the LLM Client:** Currently, business logic is tightly coupled to the OpenAI SDK. Introducing an LLMClientProtocol (Dependency Inversion) would allow injecting dummy clients for deterministic unit testing without monkey-patching, and make it trivial to swap OpenAI for Anthropic or local models.
* **Settings Management:** Replace direct os.getenv calls scattered across files with a centralized pydantic-settings configuration object to ensure environment variables are validated at startup.

### 2. Prompt & Taxonomy Centralization (DRY)
* **Externalize Prompts:** Hardcoding the system prompt inside the execution method hinders testing and version control. Prompts should be extracted into dedicated template modules.
* **Unify Taxonomy:** The incident categories and severity rubrics currently exist in prompts, tests, and documentation. Extracting these into a single configuration module will enforce DRY and prevent taxonomy drift as the system scales.

### 3. Production Resiliency & Lifecycle
* **App Lifecycle Management:** Remove import-time side effects (like logger initialization and global agent instantiation in main.py). These should be managed using FastAPI's lifespan context managers to ensure clean startup and teardown phases.
* **Resiliency & Error Handling:** Explicit timeouts and max-retry limits must be wrapped around the LLM client. Additionally, generic Exception catching should be replaced with domain-specific typed exceptions (e.g., LLMUnavailableError, LLMInvalidResponseError).
* **API Schema Expansion:** The input schema (incident_text) is too naive for real observability. Future iterations will require a richer Pydantic schema including service, timestamp, environment, and correlation_ids for end-to-end tracing.

### 4. Advanced Retrieval (RAG)
* **Deterministic Retrieval:** The current keyword-overlap approach is fragile and prone to false positives. A dedicated ContextRetriever using vector embeddings (e.g., pgvector), semantic scoring, and stop-words is required to ensure the LLM receives highly relevant historical context without context-window overflow.

### Target Architecture Structure

```text
app/
  main.py
  settings.py
  schemas.py
  agent/
    analyzer.py       # Orchestrator
    prompt_builder.py # Externalized prompt templates
    retriever.py      # Vector/Semantic retrieval
    llm_client.py     # Protocol and OpenAI implementation
    exceptions.py     # Typed domain errors
  context.py
```