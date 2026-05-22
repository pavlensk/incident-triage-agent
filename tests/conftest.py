"""
Shared pytest fixtures for the incident analysis agent test suite.

Centralising fixtures here keeps individual test modules focused on
assertions rather than setup boilerplate.
"""
import json

import pytest

from app.agent.analyzer import IncidentAnalyzer
from app.agent.retriever import ContextRetriever
from app.agent.prompt_builder import PromptBuilder


# ---------------------------------------------------------------------------
# MockLLMClient -- deterministic LLM stub
# ---------------------------------------------------------------------------

class MockLLMClient:
    """
    Deterministic stub implementing LLMClientProtocol.

    Accepts a list of pre-set response strings and returns them sequentially
    on each complete() call.  Enables fully offline, reproducible tests without
    any network interaction -- a direct benefit of Dependency Inversion.

    Raises StopIteration (propagated as a test error) if more calls are made
    than pre-set responses, making accidental over-calling visible immediately.
    """

    def __init__(self, responses: list) -> None:
        self._responses = iter(responses)
        self.call_count = 0

    async def complete(self, messages: list) -> str:
        self.call_count += 1
        return next(self._responses)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_paygate_json() -> str:
    """Canonical valid JSON response for a PayGate incident."""
    return json.dumps({
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


@pytest.fixture
def minimal_valid_json() -> str:
    """Minimal valid JSON response that satisfies all schema constraints."""
    return json.dumps({
        "category": "DB degradation",
        "severity": "medium",
        "severity_reason": "Testing self-correction recovery mechanism with sufficient reason.",
        "affected_users": "Users creating payments",
        "summary": "This is a valid summary with more than ten characters.",
        "hypotheses": [
            {
                "title": "Valid hypothesis title here",
                "reasoning": "Reasoning is long enough to pass Pydantic validation check.",
                "next_steps": [
                    "Step 1: check pg_stat_activity for long-running queries.",
                    "Step 2: review connection pool saturation metrics.",
                ],
            }
        ],
    })


def make_analyzer(
    responses: list,
    max_retries: int = 3,
    llm_retry_attempts: int = 1,
    llm_retry_delay_seconds: float = 0.0,
) -> IncidentAnalyzer:
    """
    Factory function: build an IncidentAnalyzer backed by a MockLLMClient.

    ``llm_retry_delay_seconds`` defaults to 0.0 so tests never sleep.
    ``llm_retry_attempts`` defaults to 1 (no LLM-level retry) to keep
    test behaviour predictable; set higher only when testing retry logic.

    Used directly in unit tests; also available to integration tests that need
    to override app.state.analyzer via FastAPI's dependency overrides.
    """
    return IncidentAnalyzer(
        llm_client=MockLLMClient(responses),
        retriever=ContextRetriever(),
        prompt_builder=PromptBuilder(),
        max_retries=max_retries,
        llm_retry_attempts=llm_retry_attempts,
        llm_retry_delay_seconds=llm_retry_delay_seconds,
    )
