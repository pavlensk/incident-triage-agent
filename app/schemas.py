"""
Pydantic schemas that define the data contract between the agent and the LLM.

Extracted into a dedicated module so that PromptBuilder (schema injection),
IncidentAnalyzer (response validation), and the API layer can all import
from a single authoritative source.
"""
from typing import Literal

from pydantic import BaseModel, Field


class Hypothesis(BaseModel):
    """A single hypothesis about the possible root cause of an incident."""

    title: str = Field(
        ..., min_length=5, max_length=100, description="Short title of the hypothesis"
    )
    reasoning: str = Field(
        ..., min_length=10, description="Reasoning based strictly on logs and architecture"
    )
    next_steps: list[str] = Field(
        ..., min_length=2, max_length=3, description="2-3 concrete diagnostic steps"
    )


class IncidentAnalysis(BaseModel):
    """Complete structured result of an incident analysis."""

    # Design decision: 'category' is a free-form string rather than a strict
    # Literal over TAXONOMY_CATEGORIES.  The LLM may return semantically
    # equivalent phrases that differ slightly from the canonical labels.
    # Enforcing a Literal here would cause unnecessary self-correction retries
    # for responses that are otherwise correct.  Category accuracy is validated
    # separately in tests/evals/test_taxonomy.py using gold-standard fixtures.
    category: str = Field(
        ..., min_length=3, description="Incident classification category"
    )
    severity: Literal["low", "medium", "high"] = Field(
        ..., description="Incident severity level"
    )
    severity_reason: str = Field(
        ..., min_length=10, description="Explanation for the chosen severity level"
    )
    affected_users: str = Field(
        ..., description="Who is affected (e.g. 'All card users', 'Internal team')"
    )
    summary: str = Field(
        ..., min_length=10, description="Short summary of what is happening"
    )
    hypotheses: list[Hypothesis] = Field(
        ..., min_length=1, max_length=3, description="Up to 3 hypotheses"
    )
