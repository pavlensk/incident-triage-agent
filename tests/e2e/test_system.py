"""
End-to-end system tests.

These tests simulate the five canonical incident scenarios from the assignment,
sending real HTTP requests through the full application stack:

  HTTP POST  ->  FastAPI routing  ->  input validation  ->  IncidentAnalyzer
  ->  InputParser (real)  ->  ContextRetriever (real)  ->  PromptBuilder (real)
  ->  MockLLMClient  ->  Pydantic validation  ->  HTTP 200 response

Only the LLM call is mocked (at the protocol boundary via MockLLMClient).
Every other component is the real production implementation, so these tests
catch wiring bugs that unit tests cannot see.

Design principles:
  - One test per canonical scenario from the assignment.
  - Input text is taken verbatim from the assignment to maximise realism.
  - Assertions target business requirements (correct category, severity,
    at least one hypothesis) rather than exact string matching.
  - The mock LLM response matches what a well-prompted GPT-4o-mini
    would realistically return for that scenario.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app, get_analyzer
from app.agent.analyzer import IncidentAnalyzer
from app.agent.input_parser import InputParser
from app.agent.retriever import ContextRetriever
from app.agent.prompt_builder import PromptBuilder
from tests.conftest import MockLLMClient


def _make_e2e_analyzer(llm_response: str) -> IncidentAnalyzer:
    """Build an analyzer with a single-shot mock LLM and zero retry delay."""
    return IncidentAnalyzer(
        llm_client=MockLLMClient([llm_response]),
        input_parser=InputParser(),
        retriever=ContextRetriever(),
        prompt_builder=PromptBuilder(),
        max_retries=1,
        llm_retry_attempts=1,
        llm_retry_delay_seconds=0.0,
    )


def _post_incident(client, text: str):
    return client.post("/api/v1/analyze", json={"incident_text": text})


# ---------------------------------------------------------------------------
# Scenario 1: External payment provider failure (PayGate)
# ---------------------------------------------------------------------------

def test_paygate_scenario_end_to_end(paygate_llm_response):
    """
    Scenario: Customers report card payment failures; payment-service logs show
    timeouts to PayGate.  Expected: 'External payment provider issue', high.
    """
    app.dependency_overrides[get_analyzer] = lambda: _make_e2e_analyzer(paygate_llm_response)
    try:
        with TestClient(app) as client:
            response = _post_incident(client, (
                "Customers complain that card payments often fail, "
                "and transactions do not go through.\n"
                "payment-service logs show many timeouts when calling PayGate, "
                "starting from 12:05 UTC.\n"
                "Other services look normal."
            ))
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["category"] == "External payment provider issue"
    assert body["severity"] == "high"
    assert len(body["hypotheses"]) >= 1
    assert "PayGate" in body["summary"]


# ---------------------------------------------------------------------------
# Scenario 2: Database resource saturation
# ---------------------------------------------------------------------------

def test_db_saturation_scenario_end_to_end(db_saturation_llm_response):
    """
    Scenario: Slow /payments/create; PostgreSQL shows high CPU from reporting-service.
    Expected: 'Database resource saturation', high.
    """
    app.dependency_overrides[get_analyzer] = lambda: _make_e2e_analyzer(db_saturation_llm_response)
    try:
        with TestClient(app) as client:
            response = _post_incident(client, (
                "Sharp increase in response time for /payments/create endpoint "
                "(up to 5-7 seconds).\n"
                "PostgreSQL dashboards show high CPU utilization and multiple "
                "long-running active queries from reporting-service.\n"
                "Some clients receive 504 Gateway Timeout from api-gateway."
            ))
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["category"] == "Database resource saturation"
    assert body["severity"] == "high"
    assert len(body["hypotheses"]) >= 1


# ---------------------------------------------------------------------------
# Scenario 3: Authentication token failure
# ---------------------------------------------------------------------------

def test_auth_failure_scenario_end_to_end(auth_failure_llm_response):
    """
    Scenario: Mobile users cannot log in; auth-service returns 401 with invalid
    token signatures.  Expected: 'Authentication token failure', high.
    """
    app.dependency_overrides[get_analyzer] = lambda: _make_e2e_analyzer(auth_failure_llm_response)
    try:
        with TestClient(app) as client:
            response = _post_incident(client, (
                "Mobile application users report consistent login failures.\n"
                "auth-service logs show an explicit spike in 401 Unauthorized responses.\n"
                "Internal log messages indicate invalid token signatures "
                "while other services function normally."
            ))
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["category"] == "Authentication token failure"
    assert body["severity"] == "high"
    assert len(body["hypotheses"]) >= 1


# ---------------------------------------------------------------------------
# Scenario 4: Notification delivery degradation (low severity)
# ---------------------------------------------------------------------------

def test_smtp_degradation_scenario_end_to_end(smtp_degradation_llm_response):
    """
    Scenario: Users not receiving confirmation emails; billing is fine; SMTP timeouts.
    Expected: 'Notification delivery degradation', low (financial ops intact).
    """
    app.dependency_overrides[get_analyzer] = lambda: _make_e2e_analyzer(smtp_degradation_llm_response)
    try:
        with TestClient(app) as client:
            response = _post_incident(client, (
                "Users report they are not receiving top-up confirmation emails.\n"
                "Financial balances are credited successfully and billing records "
                "are completely correct.\n"
                "notification-service logs display intermittent connection timeouts "
                "to the external SMTP provider."
            ))
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["category"] == "Notification delivery degradation"
    assert body["severity"] == "low"
    assert len(body["hypotheses"]) >= 1


# ---------------------------------------------------------------------------
# Scenario 5: Compound multi-service degradation
# ---------------------------------------------------------------------------

def test_compound_degradation_scenario_end_to_end(compound_degradation_llm_response):
    """
    Scenario: Slow payments (DB CPU) + no SMS alerts (gateway timeouts) simultaneously.
    Expected: 'Compound infrastructure degradation', high, multiple hypotheses.
    """
    app.dependency_overrides[get_analyzer] = lambda: _make_e2e_analyzer(compound_degradation_llm_response)
    try:
        with TestClient(app) as client:
            response = _post_incident(client, (
                "System performance dashboard shows payments are slow "
                "(averaging 5 seconds per request).\n"
                "Simultaneously, customer support reports users are not receiving "
                "SMS confirmations.\n"
                "Logs show severe reporting-service load on the primary DB "
                "concurrent with network timeout errors to external SMS gateways."
            ))
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["category"] == "Compound infrastructure degradation"
    assert body["severity"] == "high"
    assert len(body["hypotheses"]) == 2  # two independent root causes


# ---------------------------------------------------------------------------
# Scenario 6: Input validation guard (too short -> no LLM call)
# ---------------------------------------------------------------------------

def test_short_input_rejected_before_pipeline():
    """
    A one-liner incident report must be rejected at HTTP 400 before the
    pipeline is entered.  No LLM call should occur.
    """
    call_count = 0

    class CountingClient:
        async def complete(self, messages):
            nonlocal call_count
            call_count += 1
            return "{}"

    app.dependency_overrides[get_analyzer] = lambda: IncidentAnalyzer(
        llm_client=CountingClient(),
        input_parser=InputParser(),
        retriever=ContextRetriever(),
        prompt_builder=PromptBuilder(),
    )
    try:
        with TestClient(app) as client:
            response = client.post("/api/v1/analyze", json={"incident_text": "Too short"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 400
    assert call_count == 0, "LLM must not be called for rejected inputs"


# ---------------------------------------------------------------------------
# Scenario 7: Self-correction loop fires in the full stack
# ---------------------------------------------------------------------------

def test_self_correction_loop_end_to_end(minimal_valid_json):
    """
    First LLM response fails schema validation; second attempt succeeds.
    The full HTTP response must still be 200 -- the correction loop is
    transparent to the API caller.
    """
    invalid_first = '{"category": "incomplete"}'

    app.dependency_overrides[get_analyzer] = lambda: IncidentAnalyzer(
        llm_client=MockLLMClient([invalid_first, minimal_valid_json]),
        input_parser=InputParser(),
        retriever=ContextRetriever(),
        prompt_builder=PromptBuilder(),
        max_retries=2,
        llm_retry_attempts=1,
        llm_retry_delay_seconds=0.0,
    )
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/analyze",
                json={"incident_text": (
                    "Sharp increase in response time for /payments/create endpoint "
                    "reaching up to seven seconds per request in production logs."
                )},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["category"] == "DB degradation"
