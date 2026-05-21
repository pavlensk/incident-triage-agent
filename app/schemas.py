"""
Pydantic schemas that define the data contract between the agent and the LLM.

Extracted into a dedicated module so that PromptBuilder (schema injection),
IncidentAnalyzer (response validation), and the API layer can all import
from a single authoritative source.
"""
from typing import List, Literal

from pydantic import BaseModel, Field


class Hypothesis(BaseModel):
    """A single hypothesis about the possible root cause of an incident."""

    title: str = Field(
        ..., min_length=5, max_length=100, description="Short title of the hypothesis"
    )
    reasoning: str = Field(
        ..., min_length=10, description="Reasoning based strictly on logs and architecture"
    )
    next_steps: List[str] = Field(
        ..., min_length=2, max_length=3, description="2-3 concrete diagnostic steps"
    )


class IncidentAnalysis(BaseModel):
    """Complete structured result of an incident analysis."""

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
    hypotheses: List[Hypothesis] = Field(
        ..., min_length=1, max_length=3, description="Up to 3 hypotheses"
    )
