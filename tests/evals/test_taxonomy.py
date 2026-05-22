"""
Eval: taxonomy category accuracy.

For each of the six defined taxonomy categories, this module verifies that:
  1. The pipeline accepts a gold-standard LLM response for that category.
  2. The returned category string exactly matches the taxonomy entry.
  3. The output is schema-valid (Pydantic accepted it).

These tests act as regression guards: if a future refactoring breaks the
taxonomy contract (e.g. schema min_length tightened, category renamed in
TAXONOMY_CATEGORIES), these evals will catch it immediately.

Adding a new category:
  1. Add a gold-standard fixture to tests/evals/conftest.py.
  2. Add a new parametrize row to TestTaxonomyAccuracy below.
  3. No other changes required.
"""
import asyncio

import pytest

from app.agent.prompt_builder import TAXONOMY_CATEGORIES
from tests.conftest import make_analyzer


# Incident texts that naturally correspond to each taxonomy category.
# Kept close to real on-call language so the retrieval stage selects
# the right past incidents.

_INCIDENT_TEXTS = {
    "External payment provider issue": (
        "Customers report that card payments are failing across all regions. "
        "payment-service logs show repeated connection timeouts when calling "
        "the PayGate API starting at 14:00 UTC. Internal services are healthy."
    ),
    "Database resource saturation": (
        "Response times for /payments/create have increased to 6-8 seconds. "
        "PostgreSQL CPU is at 95% and pg_stat_activity shows dozens of "
        "long-running queries from reporting-service. Some users receive 504."
    ),
    "Authentication token failure": (
        "Mobile users cannot log in. auth-service is returning 401 errors "
        "with messages about invalid token signatures. The issue affects all "
        "mobile pods; web auth appears unaffected."
    ),
    "Notification delivery degradation": (
        "Users are not receiving top-up confirmation emails. Billing records "
        "confirm all balances were credited correctly. notification-service "
        "logs show intermittent SMTP connection timeouts."
    ),
    "Compound infrastructure degradation": (
        "Payments are slow (5+ seconds) and users report missing SMS codes. "
        "Logs show reporting-service load on the primary DB and separately "
        "network timeout errors to the SMS gateway."
    ),
    "Network routing issue": (
        "Multiple services are experiencing intermittent connectivity failures. "
        "Packet loss observed between pods in different availability zones. "
        "No single service is fully down but latency spikes to 2-3 seconds."
    ),
}


class TestTaxonomyAccuracy:
    """Each parametrize row covers one taxonomy category end-to-end."""

    @pytest.mark.parametrize("category,fixture_name", [
        ("External payment provider issue",   "eval_paygate_response"),
        ("Database resource saturation",       "eval_db_saturation_response"),
        ("Authentication token failure",       "eval_auth_failure_response"),
        ("Notification delivery degradation",  "eval_notification_degradation_response"),
        ("Compound infrastructure degradation","eval_compound_degradation_response"),
        ("Network routing issue",              "eval_network_routing_response"),
    ])
    async def test_category_matches_taxonomy(self, category, fixture_name, request):
        """Pipeline must return the correct taxonomy category for each incident type."""
        llm_response = request.getfixturevalue(fixture_name)
        incident_text = _INCIDENT_TEXTS[category]

        analyzer = make_analyzer([llm_response], max_retries=1)
        result = await analyzer.analyze(incident_text)

        assert result["category"] == category, (
            f"Expected category '{category}' but got '{result['category']}'"
        )

    def test_all_taxonomy_categories_are_covered(self):
        """Every category in TAXONOMY_CATEGORIES must have a test case above."""
        tested_categories = {cat for cat, _ in [
            ("External payment provider issue",   "eval_paygate_response"),
            ("Database resource saturation",       "eval_db_saturation_response"),
            ("Authentication token failure",       "eval_auth_failure_response"),
            ("Notification delivery degradation",  "eval_notification_degradation_response"),
            ("Compound infrastructure degradation","eval_compound_degradation_response"),
            ("Network routing issue",              "eval_network_routing_response"),
        ]}
        for cat in TAXONOMY_CATEGORIES:
            assert cat in tested_categories, (
                f"Taxonomy category '{cat}' has no eval test. Add one to test_taxonomy.py."
            )

    @pytest.mark.parametrize("category", TAXONOMY_CATEGORIES)
    async def test_schema_valid_for_each_category(self, category, request):
        """Gold-standard response for each category must pass Pydantic validation."""
        fixture_map = {
            "External payment provider issue":    "eval_paygate_response",
            "Database resource saturation":        "eval_db_saturation_response",
            "Authentication token failure":        "eval_auth_failure_response",
            "Notification delivery degradation":   "eval_notification_degradation_response",
            "Compound infrastructure degradation": "eval_compound_degradation_response",
            "Network routing issue":               "eval_network_routing_response",
        }
        fixture_name = fixture_map[category]
        llm_response = request.getfixturevalue(fixture_name)
        incident_text = _INCIDENT_TEXTS[category]

        analyzer = make_analyzer([llm_response], max_retries=1)
        result = await analyzer.analyze(incident_text)

        # If model_validate_json accepted it, these are guaranteed present
        assert "category" in result
        assert "severity" in result
        assert "hypotheses" in result
        assert isinstance(result["hypotheses"], list)
        assert len(result["hypotheses"]) >= 1
