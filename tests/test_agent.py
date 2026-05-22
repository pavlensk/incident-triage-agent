"""
Unit tests for the incident analysis agent pipeline.

Uses MockLLMClient from conftest.py -- no monkey-patching, no real API calls.
The make_analyzer() factory is also defined in conftest and available here
implicitly through pytest's conftest discovery.
"""
import json

import pytest

from app.agent.analyzer import IncidentAnalyzer
from app.agent.input_parser import InputParser
from app.agent.retriever import ContextRetriever
from app.agent.prompt_builder import PromptBuilder
from app.agent.exceptions import (
    LLMAuthenticationError,
    LLMInvalidResponseError,
    LLMRateLimitError,
    LLMUnavailableError,
)
from tests.conftest import MockLLMClient, make_analyzer


# ---------------------------------------------------------------------------
# Unit tests -- agent pipeline (happy path + validation retry)
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

    assert result.category == "External payment provider issue"
    assert result.severity == "high"
    assert "PayGate" in result.summary
    assert len(result.hypotheses) == 1

    hypothesis = result.hypotheses[0]
    assert hypothesis.title == "Degradation or incident on the PayGate side"
    assert len(hypothesis.next_steps) == 3


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
        input_parser=InputParser(),
        retriever=ContextRetriever(),
        prompt_builder=PromptBuilder(),
        max_retries=2,
        llm_retry_attempts=1,
        llm_retry_delay_seconds=0.0,
    )

    result = await analyzer.analyze(incident_text)

    assert result.category == "DB degradation"
    assert mock_client.call_count == 2, "Self-correction loop should have called the LLM twice"


async def test_failure_after_max_retries():
    """LLMInvalidResponseError must be raised once all retry attempts are exhausted."""
    incident_text = "Some users cannot log in via the mobile app due to auth failures."
    invalid_json = '{"bad": "data"}'

    analyzer = make_analyzer([invalid_json, invalid_json], max_retries=2)

    with pytest.raises(LLMInvalidResponseError, match="Failed to generate a valid response"):
        await analyzer.analyze(incident_text)


# ---------------------------------------------------------------------------
# Unit tests -- LLM-level retry (rate limit backoff)
# ---------------------------------------------------------------------------

async def test_rate_limit_retry_succeeds(valid_paygate_json):
    """
    Rate limit on the first LLM call, success on retry.

    Verifies that the exponential-backoff retry in _call_llm() fires and
    that the overall analysis still succeeds.
    """
    incident_text = (
        "Customers cannot pay by card, payment-service logs show timeouts "
        "when calling PayGate starting at 14:00 UTC."
    )

    call_count = 0

    class RateLimitThenSuccessClient:
        async def complete(self, messages: list) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise LLMRateLimitError("Rate limited on first attempt")
            return valid_paygate_json

    analyzer = IncidentAnalyzer(
        llm_client=RateLimitThenSuccessClient(),
        input_parser=InputParser(),
        retriever=ContextRetriever(),
        prompt_builder=PromptBuilder(),
        llm_retry_attempts=2,
        llm_retry_delay_seconds=0.0,  # no real sleep in tests
    )

    result = await analyzer.analyze(incident_text)

    assert result.category == "External payment provider issue"
    assert call_count == 2, "LLM should have been called exactly twice (1 rate-limited + 1 success)"


async def test_rate_limit_exhausted_raises():
    """
    Rate limit on every attempt: LLMRateLimitError must propagate after all
    retries are exhausted.  Since LLMRateLimitError IS-A LLMUnavailableError,
    catching either class works.
    """
    incident_text = (
        "Customers cannot pay by card, payment-service logs show timeouts "
        "when calling PayGate starting at 14:00 UTC."
    )

    class AlwaysRateLimitClient:
        async def complete(self, messages: list) -> str:
            raise LLMRateLimitError("Always rate limited")

    analyzer = IncidentAnalyzer(
        llm_client=AlwaysRateLimitClient(),
        input_parser=InputParser(),
        retriever=ContextRetriever(),
        prompt_builder=PromptBuilder(),
        llm_retry_attempts=2,
        llm_retry_delay_seconds=0.0,
    )

    # LLMRateLimitError inherits from LLMUnavailableError
    with pytest.raises(LLMUnavailableError):
        await analyzer.analyze(incident_text)


async def test_auth_error_propagates_immediately():
    """
    Authentication errors must propagate immediately without any retry,
    because retrying with a bad API key will never succeed.
    """
    incident_text = (
        "Customers cannot pay by card, payment-service logs show timeouts "
        "when calling PayGate starting at 14:00 UTC."
    )

    call_count = 0

    class AuthErrorClient:
        async def complete(self, messages: list) -> str:
            nonlocal call_count
            call_count += 1
            raise LLMAuthenticationError("Invalid API key")

    analyzer = IncidentAnalyzer(
        llm_client=AuthErrorClient(),
        input_parser=InputParser(),
        retriever=ContextRetriever(),
        prompt_builder=PromptBuilder(),
        llm_retry_attempts=3,       # high retry count -- should NOT be used
        llm_retry_delay_seconds=0.0,
    )

    with pytest.raises(LLMAuthenticationError):
        await analyzer.analyze(incident_text)

    assert call_count == 1, "Authentication error must not trigger any retry"


# ---------------------------------------------------------------------------
# Unit tests -- InputParser
# ---------------------------------------------------------------------------

def test_input_parser_selects_smtp_incident():
    """SMTP / email query should produce keywords that match INC-103, not INC-101 (PayGate)."""
    parser = InputParser()
    retriever = ContextRetriever()
    parsed = parser.parse_input("Users are not receiving emails from smtp provider")
    context = retriever.retrieve(parsed)
    assert "SMTP provider" in context
    assert "PayGate provider" not in context


def test_input_parser_selects_db_incident():
    """DB / reporting load query should produce keywords that match INC-102."""
    parser = InputParser()
    retriever = ContextRetriever()
    parsed = parser.parse_input("CPU load is high on PostgreSQL due to reporting queries")
    context = retriever.retrieve(parsed)
    assert "reporting-service" in context
    assert "DB dashboards show high CPU" in context


def test_input_parser_selects_auth_incident():
    """Auth failure query should produce keywords matching INC-104."""
    parser = InputParser()
    retriever = ContextRetriever()
    parsed = parser.parse_input("auth service returns 401 errors with invalid token signatures")
    context = retriever.retrieve(parsed)
    assert "401" in context or "token" in context.lower()


def test_retriever_fallback_on_no_match():
    """When no incidents match, the retriever must return the first two as a fallback."""
    parser = InputParser()
    retriever = ContextRetriever()
    parsed = parser.parse_input("xyzzy qwerty frobnicate something unknown")
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
