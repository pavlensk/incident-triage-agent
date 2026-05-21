"""
Tests for the incident analysis agent.

Key benefit of the new architecture: tests use MockLLMClient -- a simple stub
implementing LLMClientProtocol. No monkey-patching, no AsyncMock over openai
SDK internals. This is a direct consequence of the Dependency Inversion
Principle: the dependency is injected from outside rather than created inside
the component under test.
"""
import json

import pytest

from app.agent.analyzer import IncidentAnalyzer
from app.agent.retriever import ContextRetriever
from app.agent.prompt_builder import PromptBuilder
from app.agent.exceptions import LLMInvalidResponseError


# ---------------------------------------------------------------------------
# LLM client stub
# ---------------------------------------------------------------------------

class MockLLMClient:
    """
    Deterministic stub implementing LLMClientProtocol.

    Accepts a list of strings and returns them sequentially on each complete()
    call. Enables testing the 'first response invalid, second valid' path
    without any network interaction.
    """

    def __init__(self, responses: list) -> None:
        self._responses = iter(responses)
        self.call_count = 0

    async def complete(self, messages: list) -> str:
        self.call_count += 1
        return next(self._responses)


# ---------------------------------------------------------------------------
# Test factory
# ---------------------------------------------------------------------------

def make_analyzer(responses: list, max_retries: int = 3) -> IncidentAnalyzer:
    """Create an IncidentAnalyzer with a stub LLM and standard components."""
    return IncidentAnalyzer(
        llm_client=MockLLMClient(responses),
        retriever=ContextRetriever(),
        prompt_builder=PromptBuilder(),
        max_retries=max_retries,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_happy_path_canonical_example():
    """Successful analysis of the canonical example from the assignment."""
    incident_text = (
        "Customers complain that card payments often fail, and transactions do not go through.\n"
        "payment-service logs show many timeouts when calling PayGate, starting from 12:05 UTC.\n"
        "Other services look normal."
    )

    valid_json = json.dumps({
        "category": "External payment provider issue",
        "severity": "high",
        "severity_reason": "Massive payment failures directly impact revenue.",
        "affected_users": "All customers attempting card payments",
        "summary": "The external provider PayGate is not responding in time, causing mass card payment failures.",
        "hypotheses": [
            {
                "title": "Degradation or incident on the PayGate side",
                "reasoning": "Timeouts are observed only when calling PayGate, other services remain stable.",
                "next_steps": [
                    "Check PayGate status page and recent provider notifications.",
                    "Compare error and latency metrics for PayGate vs other payment providers.",
                    "Temporarily shift part of the traffic to an alternative provider.",
                ],
            }
        ],
    })

    analyzer = make_analyzer([valid_json])
    result = await analyzer.analyze(incident_text)

    assert result["category"] == "External payment provider issue"
    assert result["severity"] == "high"
    assert "PayGate" in result["summary"]
    assert len(result["hypotheses"]) == 1

    hypothesis = result["hypotheses"][0]
    assert hypothesis["title"] == "Degradation or incident on the PayGate side"
    assert len(hypothesis["next_steps"]) == 3


@pytest.mark.asyncio
async def test_retry_on_invalid_json():
    """
    Self-correction loop: first response is invalid -> retry -> success.

    Also verifies that MockLLMClient was called exactly twice,
    confirming the correction loop actually fired.
    """
    incident_text = "Sharp increase in response time for /payments/create (up to 5-7 seconds)."

    invalid_json = '{"category": "Missing required fields"}'
    valid_json = json.dumps({
        "category": "DB degradation",
        "severity": "medium",
        "severity_reason": "Testing self-correction recovery mechanism with sufficient reason.",
        "affected_users": "Users creating payments",
        "summary": "This is a valid summary with more than ten characters.",
        "hypotheses": [
            {
                "title": "Valid hypothesis title here",
                "reasoning": "Reasoning is long enough to pass Pydantic validation check.",
                "next_steps": ["Step 1: check pg_stat_activity", "Step 2: review connection pool"],
            }
        ],
    })

    mock_client = MockLLMClient([invalid_json, valid_json])
    analyzer = IncidentAnalyzer(
        llm_client=mock_client,
        retriever=ContextRetriever(),
        prompt_builder=PromptBuilder(),
        max_retries=2,
    )

    result = await analyzer.analyze(incident_text)

    assert result["category"] == "DB degradation"
    assert mock_client.call_count == 2, "Self-correction loop should have called the LLM twice"


@pytest.mark.asyncio
async def test_failure_after_max_retries():
    """LLMInvalidResponseError must be raised once all retry attempts are exhausted."""
    incident_text = "Some users cannot log in via the mobile app due to auth failures."
    invalid_json = '{"bad": "data"}'

    analyzer = make_analyzer([invalid_json, invalid_json], max_retries=2)

    with pytest.raises(LLMInvalidResponseError, match="Failed to generate a valid response"):
        await analyzer.analyze(incident_text)


def test_retriever_selects_relevant_incidents():
    """ContextRetriever should select semantically appropriate past incidents."""
    retriever = ContextRetriever()

    # SMTP / email query -> should return INC-103, not INC-101
    smtp_parsed = retriever.parse_input("Users are not receiving emails from smtp provider")
    smtp_context = retriever.retrieve(smtp_parsed)
    assert "SMTP provider" in smtp_context
    assert "PayGate provider" not in smtp_context

    # DB load query -> should return INC-102
    db_parsed = retriever.parse_input("CPU load is high on PostgreSQL due to reporting queries")
    db_context = retriever.retrieve(db_parsed)
    assert "reporting-service" in db_context
    assert "DB dashboards show high CPU" in db_context


def test_retriever_fallback_on_no_match():
    """When no incidents match, the retriever should return the first two as a fallback."""
    retriever = ContextRetriever()
    parsed = retriever.parse_input("xyzzy qwerty frobnicate something unknown")
    context = retriever.retrieve(parsed)
    assert "INC-101" in context or "INC-102" in context


def test_prompt_builder_contains_schema_and_architecture():
    """The system prompt must include the JSON schema, architecture, and severity rules."""
    builder = PromptBuilder()
    prompt = builder.build_system_prompt("Test context")

    assert "json" in prompt.lower()      # schema is present
    assert "api-gateway" in prompt       # architecture is present
    assert "Test context" in prompt      # RAG context is embedded
    assert "severity" in prompt          # severity rules are present
