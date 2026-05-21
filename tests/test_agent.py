"""
Unit tests for the incident analysis agent pipeline.

Uses MockLLMClient from conftest.py -- no monkey-patching, no real API calls.
The make_analyzer() factory is also defined in conftest and available here
implicitly through pytest's conftest discovery.
"""
import json

import pytest

from app.agent.analyzer import IncidentAnalyzer
from app.agent.retriever import ContextRetriever
from app.agent.prompt_builder import PromptBuilder
from app.agent.exceptions import LLMInvalidResponseError
from tests.conftest import MockLLMClient, make_analyzer


# ---------------------------------------------------------------------------
# Unit tests -- agent pipeline
# ---------------------------------------------------------------------------

async def test_happy_path_canonical_example(valid_paygate_json):
    """Successful analysis of the canonical example from the assignment."""
    incident_text = (
        "Customers complain that card payments often fail, and transactions do not go through.\n"
        "payment-service logs show many timeouts when calling PayGate, starting from 12:05 UTC.\n"
        "Other services look normal."
    )

    analyzer = make_analyzer([valid_paygate_json])
    result = await analyzer.analyze(incident_text)

    assert result["category"] == "External payment provider issue"
    assert result["severity"] == "high"
    assert "PayGate" in result["summary"]
    assert len(result["hypotheses"]) == 1

    hypothesis = result["hypotheses"][0]
    assert hypothesis["title"] == "Degradation or incident on the PayGate side"
    assert len(hypothesis["next_steps"]) == 3


async def test_retry_on_invalid_json(minimal_valid_json):
    """
    Self-correction loop: first response is invalid -> retry -> success.

    Verifies that MockLLMClient was called exactly twice, confirming
    the correction loop actually fired.
    """
    incident_text = "Sharp increase in response time for /payments/create (up to 5-7 seconds)."
    invalid_json = '{"category": "Missing required fields"}'

    mock_client = MockLLMClient([invalid_json, minimal_valid_json])
    analyzer = IncidentAnalyzer(
        llm_client=mock_client,
        retriever=ContextRetriever(),
        prompt_builder=PromptBuilder(),
        max_retries=2,
    )

    result = await analyzer.analyze(incident_text)

    assert result["category"] == "DB degradation"
    assert mock_client.call_count == 2, "Self-correction loop should have called the LLM twice"


async def test_failure_after_max_retries():
    """LLMInvalidResponseError must be raised once all retry attempts are exhausted."""
    incident_text = "Some users cannot log in via the mobile app due to auth failures."
    invalid_json = '{"bad": "data"}'

    analyzer = make_analyzer([invalid_json, invalid_json], max_retries=2)

    with pytest.raises(LLMInvalidResponseError, match="Failed to generate a valid response"):
        await analyzer.analyze(incident_text)


# ---------------------------------------------------------------------------
# Unit tests -- ContextRetriever
# ---------------------------------------------------------------------------

def test_retriever_selects_smtp_incident():
    """SMTP / email query should return INC-103, not INC-101 (PayGate)."""
    retriever = ContextRetriever()
    parsed = retriever.parse_input("Users are not receiving emails from smtp provider")
    context = retriever.retrieve(parsed)
    assert "SMTP provider" in context
    assert "PayGate provider" not in context


def test_retriever_selects_db_incident():
    """DB / reporting load query should return INC-102."""
    retriever = ContextRetriever()
    parsed = retriever.parse_input("CPU load is high on PostgreSQL due to reporting queries")
    context = retriever.retrieve(parsed)
    assert "reporting-service" in context
    assert "DB dashboards show high CPU" in context


def test_retriever_selects_auth_incident():
    """Auth failure query should return INC-104 -- validates fix for len > 2 threshold."""
    retriever = ContextRetriever()
    parsed = retriever.parse_input("auth service returns 401 errors with invalid token signatures")
    context = retriever.retrieve(parsed)
    assert "401" in context or "token" in context.lower()


def test_retriever_fallback_on_no_match():
    """When no incidents match, the retriever must return the first two as a fallback."""
    retriever = ContextRetriever()
    parsed = retriever.parse_input("xyzzy qwerty frobnicate something unknown")
    context = retriever.retrieve(parsed)
    assert "INC-101" in context or "INC-102" in context


# ---------------------------------------------------------------------------
# Unit tests -- PromptBuilder
# ---------------------------------------------------------------------------

def test_prompt_builder_contains_schema_and_architecture():
    """The system prompt must include the JSON schema, architecture, and severity rules."""
    builder = PromptBuilder()
    prompt = builder.build_system_prompt("Test context")

    assert "json" in prompt.lower()   # schema is present
    assert "api-gateway" in prompt    # architecture is present
    assert "Test context" in prompt   # RAG context is embedded
    assert "severity" in prompt       # severity rules are present


def test_prompt_builder_correction_message_contains_errors():
    """Correction message must pass through the Pydantic error details."""
    builder = PromptBuilder()
    msg = builder.build_correction_message('{"error": "field required"}')
    assert "field required" in msg
    assert "Pydantic" in msg
