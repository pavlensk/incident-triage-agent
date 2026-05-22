"""
Integration tests for the FastAPI HTTP layer.

Uses FastAPI's TestClient with dependency_overrides to inject mock LLM clients,
so tests remain offline and deterministic while covering the full HTTP stack:
routing, request validation, response codes, error mapping, and response format.

Error response format for domain exceptions:
    {"code": "<machine_readable>", "message": "<human_readable>"}

HTTPException responses (400 / Pydantic 422) keep FastAPI's default format:
    {"detail": "..."}
"""
import json

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_analyzer
from app.agent.analyzer import IncidentAnalyzer
from app.agent.retriever import ContextRetriever
from app.agent.prompt_builder import PromptBuilder
from app.agent.exceptions import (
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMUnavailableError,
)
from tests.conftest import MockLLMClient, make_analyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _override_analyzer(responses: list, max_retries: int = 3):
    """
    Return a FastAPI dependency override that injects a mock-backed analyzer.

    Usage:
        app.dependency_overrides[get_analyzer] = _override_analyzer([json_str])
    """
    analyzer = make_analyzer(responses, max_retries=max_retries)

    def _get() -> object:
        return analyzer

    return _get


def _override_with(analyzer: IncidentAnalyzer):
    """Return a dependency override that returns a pre-built analyzer."""
    return lambda: analyzer


def _make_stub_analyzer(client_class) -> IncidentAnalyzer:
    """Build an IncidentAnalyzer backed by an arbitrary stub LLM client."""
    return IncidentAnalyzer(
        llm_client=client_class(),
        retriever=ContextRetriever(),
        prompt_builder=PromptBuilder(),
        llm_retry_attempts=1,       # no retries -- tests must be deterministic
        llm_retry_delay_seconds=0.0,
    )


# ---------------------------------------------------------------------------
# Integration tests -- success path
# ---------------------------------------------------------------------------

def test_analyze_returns_200_with_valid_payload(valid_paygate_json):
    """A well-formed incident description must return HTTP 200 with structured JSON."""
    app.dependency_overrides[get_analyzer] = _override_analyzer([valid_paygate_json])

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/analyze",
            json={"incident_text": (
                "Customers complain that card payments often fail. "
                "payment-service logs show timeouts when calling PayGate starting at 12:05 UTC. "
                "Other services look normal."
            )},
        )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["category"] == "External payment provider issue"
    assert body["severity"] == "high"
    assert "hypotheses" in body
    assert len(body["hypotheses"]) >= 1


# ---------------------------------------------------------------------------
# Integration tests -- input validation errors
# ---------------------------------------------------------------------------

def test_analyze_returns_400_for_too_short_text():
    """Incident text that is too short must be rejected with HTTP 400 before hitting the LLM."""
    app.dependency_overrides[get_analyzer] = _override_analyzer([])

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/analyze",
            json={"incident_text": "Too short"},
        )

    app.dependency_overrides.clear()

    assert response.status_code == 400
    assert "too short" in response.json()["detail"].lower()


def test_analyze_returns_422_for_missing_body_field():
    """A request missing the required incident_text field must return HTTP 422 (Pydantic)."""
    with TestClient(app) as client:
        response = client.post("/api/v1/analyze", json={})

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Integration tests -- domain exception -> HTTP code mapping
# ---------------------------------------------------------------------------

def test_analyze_returns_422_when_llm_exhausts_retries():
    """When the LLM returns invalid JSON on all retries, the API must respond with HTTP 422."""
    always_invalid = '{"bad": "data"}'
    app.dependency_overrides[get_analyzer] = _override_analyzer(
        [always_invalid, always_invalid], max_retries=2
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/analyze",
            json={"incident_text": (
                "Sharp increase in response time for the payments endpoint "
                "reaching up to seven seconds per request observed in logs."
            )},
        )

    app.dependency_overrides.clear()

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "llm_invalid_response"
    assert "message" in body


def test_analyze_returns_503_when_llm_unavailable():
    """When the LLM API is unreachable, the endpoint must return HTTP 503."""

    class UnavailableLLMClient:
        """Stub that always raises LLMUnavailableError."""
        async def complete(self, messages: list) -> str:
            raise LLMUnavailableError("Connection refused")

    app.dependency_overrides[get_analyzer] = _override_with(
        _make_stub_analyzer(UnavailableLLMClient)
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/analyze",
            json={"incident_text": (
                "Mobile users cannot log in, auth-service returns 401 errors "
                "and logs show invalid token signatures across all pods."
            )},
        )

    app.dependency_overrides.clear()

    assert response.status_code == 503
    body = response.json()
    assert body["code"] == "llm_unavailable"
    assert "message" in body


def test_analyze_returns_503_when_rate_limited():
    """Rate-limit errors must surface as HTTP 503 (LLMRateLimitError IS-A LLMUnavailableError)."""

    class RateLimitClient:
        """Stub that always raises LLMRateLimitError."""
        async def complete(self, messages: list) -> str:
            raise LLMRateLimitError("Rate limit exceeded")

    app.dependency_overrides[get_analyzer] = _override_with(
        _make_stub_analyzer(RateLimitClient)
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/analyze",
            json={"incident_text": (
                "Mobile users cannot log in, auth-service returns 401 errors "
                "and logs show invalid token signatures across all pods."
            )},
        )

    app.dependency_overrides.clear()

    assert response.status_code == 503
    body = response.json()
    assert body["code"] == "llm_unavailable"


def test_analyze_returns_500_on_auth_error():
    """
    An invalid / revoked API key must return HTTP 500 with a safe, non-leaking
    message -- it is a server-side configuration problem, not a client error.
    """

    class AuthErrorClient:
        """Stub that always raises LLMAuthenticationError."""
        async def complete(self, messages: list) -> str:
            raise LLMAuthenticationError("Invalid API key")

    app.dependency_overrides[get_analyzer] = _override_with(
        _make_stub_analyzer(AuthErrorClient)
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/analyze",
            json={"incident_text": (
                "Mobile users cannot log in, auth-service returns 401 errors "
                "and logs show invalid token signatures across all pods."
            )},
        )

    app.dependency_overrides.clear()

    assert response.status_code == 500
    body = response.json()
    assert body["code"] == "llm_configuration_error"
    # The response must NOT leak the raw API key error -- only a safe message.
    assert "misconfigured" in body["message"].lower()


# ---------------------------------------------------------------------------
# Integration tests -- static frontend
# ---------------------------------------------------------------------------

def test_static_frontend_is_served():
    """The root path must serve the static HTML frontend."""
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
