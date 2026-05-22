"""
Unit tests for Pydantic data contracts (schemas.py).

These tests validate every field constraint in isolation -- min/max lengths,
literal enumerations, and list size bounds -- so that changes to the schema
surface immediately as failing tests rather than silent runtime errors.

Boundary-value testing principle:
  - valid at lower bound  (passes)
  - valid at upper bound  (passes)
  - one below lower bound (raises ValidationError)
  - one above upper bound (raises ValidationError)
"""
import json

import pytest
from pydantic import ValidationError

from app.schemas import Hypothesis, IncidentAnalysis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hypothesis(**overrides) -> dict:
    """Return a valid Hypothesis payload, with selective field overrides."""
    base = {
        "title": "Valid title here",        # 16 chars, >= 5 ✓
        "reasoning": "Long enough reasoning text.",  # >= 10 ✓
        "next_steps": [
            "Step one: check logs.",
            "Step two: check metrics.",
        ],
    }
    base.update(overrides)
    return base


def _make_analysis(**overrides) -> dict:
    """Return a valid IncidentAnalysis payload, with selective field overrides."""
    base = {
        "category": "External payment provider issue",
        "severity": "high",
        "severity_reason": "Core payment operations are fully blocked.",
        "affected_users": "All card users",
        "summary": "PayGate timeouts blocking all payments.",
        "hypotheses": [_make_hypothesis()],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Hypothesis field constraints
# ---------------------------------------------------------------------------

class TestHypothesisTitle:
    def test_title_at_min_length_passes(self):
        h = Hypothesis(**_make_hypothesis(title="12345"))  # exactly 5 chars
        assert h.title == "12345"

    def test_title_at_max_length_passes(self):
        h = Hypothesis(**_make_hypothesis(title="x" * 100))  # exactly 100
        assert len(h.title) == 100

    def test_title_below_min_raises(self):
        with pytest.raises(ValidationError):
            Hypothesis(**_make_hypothesis(title="1234"))  # 4 chars < 5

    def test_title_above_max_raises(self):
        with pytest.raises(ValidationError):
            Hypothesis(**_make_hypothesis(title="x" * 101))  # 101 > 100


class TestHypothesisReasoning:
    def test_reasoning_at_min_length_passes(self):
        h = Hypothesis(**_make_hypothesis(reasoning="1234567890"))  # exactly 10
        assert len(h.reasoning) == 10

    def test_reasoning_below_min_raises(self):
        with pytest.raises(ValidationError):
            Hypothesis(**_make_hypothesis(reasoning="short"))  # 5 < 10


class TestHypothesisNextSteps:
    def test_two_steps_passes(self):
        h = Hypothesis(**_make_hypothesis(next_steps=["step a.", "step b."]))
        assert len(h.next_steps) == 2

    def test_three_steps_passes(self):
        h = Hypothesis(**_make_hypothesis(next_steps=["a", "b", "c"]))
        assert len(h.next_steps) == 3

    def test_one_step_raises(self):
        with pytest.raises(ValidationError):
            Hypothesis(**_make_hypothesis(next_steps=["only one step."]))

    def test_four_steps_raises(self):
        with pytest.raises(ValidationError):
            Hypothesis(**_make_hypothesis(next_steps=["a", "b", "c", "d"]))

    def test_empty_steps_raises(self):
        with pytest.raises(ValidationError):
            Hypothesis(**_make_hypothesis(next_steps=[]))


# ---------------------------------------------------------------------------
# IncidentAnalysis field constraints
# ---------------------------------------------------------------------------

class TestSeverityLiteral:
    @pytest.mark.parametrize("severity", ["low", "medium", "high"])
    def test_valid_severity_values(self, severity):
        a = IncidentAnalysis(**_make_analysis(severity=severity))
        assert a.severity == severity

    def test_invalid_severity_raises(self):
        with pytest.raises(ValidationError):
            IncidentAnalysis(**_make_analysis(severity="critical"))

    def test_uppercase_severity_raises(self):
        with pytest.raises(ValidationError):
            IncidentAnalysis(**_make_analysis(severity="High"))


class TestIncidentAnalysisCategory:
    def test_category_at_min_length_passes(self):
        a = IncidentAnalysis(**_make_analysis(category="abc"))  # 3 chars = min
        assert a.category == "abc"

    def test_category_below_min_raises(self):
        with pytest.raises(ValidationError):
            IncidentAnalysis(**_make_analysis(category="ab"))  # 2 < 3


class TestIncidentAnalysisHypotheses:
    def test_one_hypothesis_passes(self):
        a = IncidentAnalysis(**_make_analysis(hypotheses=[_make_hypothesis()]))
        assert len(a.hypotheses) == 1

    def test_three_hypotheses_passes(self):
        a = IncidentAnalysis(**_make_analysis(
            hypotheses=[_make_hypothesis(), _make_hypothesis(), _make_hypothesis()]
        ))
        assert len(a.hypotheses) == 3

    def test_zero_hypotheses_raises(self):
        with pytest.raises(ValidationError):
            IncidentAnalysis(**_make_analysis(hypotheses=[]))

    def test_four_hypotheses_raises(self):
        with pytest.raises(ValidationError):
            IncidentAnalysis(**_make_analysis(
                hypotheses=[_make_hypothesis()] * 4
            ))


class TestIncidentAnalysisSummary:
    def test_summary_at_min_length_passes(self):
        a = IncidentAnalysis(**_make_analysis(summary="1234567890"))  # 10 chars
        assert len(a.summary) == 10

    def test_summary_below_min_raises(self):
        with pytest.raises(ValidationError):
            IncidentAnalysis(**_make_analysis(summary="short"))


class TestIncidentAnalysisSeverityReason:
    def test_severity_reason_at_min_length_passes(self):
        a = IncidentAnalysis(**_make_analysis(severity_reason="1234567890"))
        assert len(a.severity_reason) == 10

    def test_severity_reason_below_min_raises(self):
        with pytest.raises(ValidationError):
            IncidentAnalysis(**_make_analysis(severity_reason="too short"))


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------

class TestJsonRoundTrip:
    def test_model_to_json_and_back(self):
        """model_dump() -> JSON -> model_validate_json() must produce identical data."""
        original = IncidentAnalysis(**_make_analysis())
        json_str = json.dumps(original.model_dump())
        restored = IncidentAnalysis.model_validate_json(json_str)
        assert original.model_dump() == restored.model_dump()

    def test_invalid_json_raises_validation_error(self):
        with pytest.raises(ValidationError):
            IncidentAnalysis.model_validate_json("not valid json at all")

    def test_empty_json_object_raises_validation_error(self):
        with pytest.raises(ValidationError):
            IncidentAnalysis.model_validate_json("{}")
