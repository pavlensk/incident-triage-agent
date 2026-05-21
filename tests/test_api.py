"""
Integration tests for the FastAPI HTTP layer.

Uses FastAPI's TestClient with dependency_overrides to inject a MockLLMClient,
so tests remain offline and deterministic while covering the full HTTP stack:
routing, request validation, response codes, and error mapping.
"""
import json

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_analyzer
from app.agent.exceptions import LLMUnavailableError
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


# ---------------------------------------------------------------------------
# Integration tests
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


def test_analyze_returns_503_when_llm_unavailable():
    """When the LLM API is unreachable, the endpoint must return HTTP 503."""

    class UnavailableLLMClient:
        """Stub that always raises LLMUnavailableError."""
        async def complete(self, messages: list) -> str:
            raise LLMUnavailableError("Connection refused")

    from app.agent.analyzer import IncidentAnalyzer
    from app.agent.retriever import ContextRetriever
    from app.agent.prompt_builder import PromptBuilder

    unavailable_analyzer = IncidentAnalyzer(
        llm_client=UnavailableLLMClient(),
        retriever=ContextRetriever(),
        prompt_builder=PromptBuilder(),
    )
    app.dependency_overrides[get_analyzer] = lambda: unavailable_analyzer

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


def test_analyze_returns_422_for_missing_body_field():
    """A request missing the required incident_text field must return HTTP 422 (Pydantic)."""
    with TestClient(app) as client:
        response = client.post("/api/v1/analyze", json={})

    assert response.status_code == 422


def test_static_frontend_is_served():
    """The root path must serve the static HTML frontend."""
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
