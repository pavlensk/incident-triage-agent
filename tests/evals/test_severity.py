"""
Eval: severity rubric adherence.

The severity rubric defines three levels:
  high   -- core financial operations or authentication blocked;
             mass payment failures; DB saturation with cascading timeouts.
  medium -- authentication degraded for subset of users;
             critical MFA SMS delayed.
  low    -- financial operations fully intact;
             confirmation notification delayed.

These tests verify that:
  1. Gold-standard LLM responses carry the correct severity for each scenario.
  2. The pipeline passes responses with each valid severity through validation.
  3. Invalid severity values are rejected before reaching the caller.

Severity rubric reference: app/agent/prompt_builder.py::SEVERITY_RUBRIC
"""
import pytest
from pydantic import ValidationError

from app.schemas import IncidentAnalysis
from tests.conftest import make_analyzer


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _parse(json_str: str) -> IncidentAnalysis:
    return IncidentAnalysis.model_validate_json(json_str)


# ---------------------------------------------------------------------------
# Rubric adherence: each gold-standard response carries the right severity
# ---------------------------------------------------------------------------

class TestSeverityRubricAdherence:
    """
    Verify that every gold-standard fixture assigns severity consistent
    with the rubric defined in SEVERITY_RUBRIC.
    """

    def test_paygate_is_high(self, eval_paygate_response):
        """Mass payment failures → high."""
        result = _parse(eval_paygate_response)
        assert result.severity == "high", (
            "PayGate outage blocks core financial operations -- must be 'high'"
        )

    def test_db_saturation_is_high(self, eval_db_saturation_response):
        """DB saturation with cascading payment timeouts → high."""
        result = _parse(eval_db_saturation_response)
        assert result.severity == "high"

    def test_auth_failure_is_high(self, eval_auth_failure_response):
        """Authentication fully blocked → high."""
        result = _parse(eval_auth_failure_response)
        assert result.severity == "high"

    def test_smtp_degradation_is_low(self, eval_notification_degradation_response):
        """Financial operations intact, only notification delayed → low."""
        result = _parse(eval_notification_degradation_response)
        assert result.severity == "low", (
            "Delayed notification with intact billing must be 'low' per rubric"
        )

    def test_compound_degradation_is_high(self, eval_compound_degradation_response):
        """Payment processing affected → high regardless of secondary issues."""
        result = _parse(eval_compound_degradation_response)
        assert result.severity == "high"

    def test_network_routing_is_medium(self, eval_network_routing_response):
        """Intermittent degradation without full blockage → medium."""
        result = _parse(eval_network_routing_response)
        assert result.severity == "medium"


# ---------------------------------------------------------------------------
# Rubric boundary: all three valid values pass Pydantic validation
# ---------------------------------------------------------------------------

class TestSeverityValidation:
    @pytest.mark.parametrize("severity", ["low", "medium", "high"])
    async def test_all_valid_severity_values_accepted_by_pipeline(
        self, severity, eval_notification_degradation_response
    ):
        """
        Rewrite the severity in a gold-standard response and confirm
        the pipeline still accepts it for all three valid values.
        """
        import json
        payload = json.loads(eval_notification_degradation_response)
        payload["severity"] = severity
        # Adjust reason to be plausible for each severity
        payload["severity_reason"] = f"Testing severity={severity} acceptance in pipeline."

        analyzer = make_analyzer([json.dumps(payload)], max_retries=1)
        result = await analyzer.analyze(
            "Users are not receiving confirmation emails after top-up. "
            "Billing records are correct and all payments succeed normally."
        )
        assert result["severity"] == severity

    def test_invalid_severity_rejected_by_schema(self):
        """
        A severity value outside the literal set must raise ValidationError
        before any business logic runs.
        """
        import json
        payload = {
            "category": "External payment provider issue",
            "severity": "critical",  # not in Literal["low", "medium", "high"]
            "severity_reason": "All payments blocked.",
            "affected_users": "All card users",
            "summary": "PayGate is down.",
            "hypotheses": [{
                "title": "PayGate outage",
                "reasoning": "Timeouts only when calling PayGate.",
                "next_steps": ["Check status page.", "Switch gateway."],
            }],
        }
        with pytest.raises(ValidationError):
            IncidentAnalysis.model_validate_json(json.dumps(payload))

    def test_severity_reason_required_and_nonempty(self):
        """severity_reason must be present and at least 10 characters."""
        import json
        payload = {
            "category": "External payment provider issue",
            "severity": "high",
            "severity_reason": "short",   # 5 chars < 10
            "affected_users": "All card users",
            "summary": "PayGate timeouts.",
            "hypotheses": [{
                "title": "PayGate outage",
                "reasoning": "Timeouts only on PayGate calls.",
                "next_steps": ["Check status page.", "Switch gateway."],
            }],
        }
        with pytest.raises(ValidationError):
            IncidentAnalysis.model_validate_json(json.dumps(payload))
