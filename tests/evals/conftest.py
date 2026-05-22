"""
Eval fixtures: gold-standard expected outputs for all six taxonomy categories.

In this offline eval suite, "gold standard" means a pre-recorded, schema-valid
LLM response that represents the correct answer for a known incident input.

How evals differ from unit/integration tests:
  - Unit/integration tests verify MECHANICAL CORRECTNESS (does the code run?).
  - Eval tests verify OUTPUT QUALITY (does the system produce the right answer?).

Extending evals:
  Add a new (incident_text, expected_category, expected_severity, llm_response)
  tuple to the appropriate parametrize list in the test files.  The fixtures
  here provide the LLM mock responses; actual inputs live in the test files
  alongside the expected assertions.
"""
import json
import pytest


# ---------------------------------------------------------------------------
# Gold-standard LLM responses per taxonomy category
# ---------------------------------------------------------------------------

def _make_response(
    category: str,
    severity: str,
    severity_reason: str,
    summary: str,
    hypothesis_title: str,
    hypothesis_reasoning: str,
    next_steps: list,
    affected_users: str = "Affected platform users",
    extra_hypotheses: list | None = None,
) -> str:
    hypotheses = [{
        "title": hypothesis_title,
        "reasoning": hypothesis_reasoning,
        "next_steps": next_steps,
    }]
    if extra_hypotheses:
        hypotheses.extend(extra_hypotheses)
    return json.dumps({
        "category": category,
        "severity": severity,
        "severity_reason": severity_reason,
        "affected_users": affected_users,
        "summary": summary,
        "hypotheses": hypotheses,
    })


@pytest.fixture
def eval_paygate_response() -> str:
    return _make_response(
        category="External payment provider issue",
        severity="high",
        severity_reason="All card payments are blocked; direct revenue loss.",
        summary="PayGate timeouts blocking all outbound card payment processing.",
        hypothesis_title="PayGate service degradation",
        hypothesis_reasoning=(
            "Timeouts are isolated to PayGate API calls; internal services remain stable."
        ),
        next_steps=[
            "Check PayGate public status page.",
            "Switch traffic to backup payment gateway.",
            "Alert on-call PayGate account manager.",
        ],
        affected_users="All card-paying customers",
    )


@pytest.fixture
def eval_db_saturation_response() -> str:
    return _make_response(
        category="Database resource saturation",
        severity="high",
        severity_reason=(
            "DB CPU saturation causes cascading payment timeouts and 504 errors."
        ),
        summary=(
            "reporting-service queries saturating PostgreSQL CPU, "
            "blocking payment-service transactions."
        ),
        hypothesis_title="Analytical queries exhausting DB CPU",
        hypothesis_reasoning=(
            "pg_stat_activity shows long-running reporting queries on the primary DB "
            "running concurrently with payment transactions."
        ),
        next_steps=[
            "Kill long-running queries via pg_stat_activity.",
            "Redirect reporting-service to a read replica.",
            "Throttle or pause reporting jobs during peak hours.",
        ],
        affected_users="Users creating payment transactions",
    )


@pytest.fixture
def eval_auth_failure_response() -> str:
    return _make_response(
        category="Authentication token failure",
        severity="high",
        severity_reason="All mobile users are blocked from logging in.",
        summary="auth-service rejecting tokens with invalid signature errors on all pods.",
        hypothesis_title="JWT signing key mismatch after secret rotation",
        hypothesis_reasoning=(
            "Invalid signatures across all pods imply a signing/verification key "
            "mismatch, likely caused by a recent key rotation not applied uniformly."
        ),
        next_steps=[
            "Inspect Vault for recent key rotation events.",
            "Verify JWT public key across all auth-service pods.",
            "Roll back to the previous key if mismatch is confirmed.",
        ],
        affected_users="All mobile app users",
    )


@pytest.fixture
def eval_notification_degradation_response() -> str:
    return _make_response(
        category="Notification delivery degradation",
        severity="low",
        severity_reason=(
            "Financial operations are intact; only non-critical email delivery is affected."
        ),
        summary="notification-service failing to reach SMTP provider; emails are delayed.",
        hypothesis_title="SMTP provider outage or routing issue",
        hypothesis_reasoning=(
            "Connection timeouts to SMTP are isolated; billing confirms money credited fine."
        ),
        next_steps=[
            "Check SMTP provider status page.",
            "Monitor the notification retry queue for backlog growth.",
            "Verify cluster outbound network connectivity to the SMTP endpoint.",
        ],
        affected_users="Users expecting confirmation emails",
    )


@pytest.fixture
def eval_compound_degradation_response() -> str:
    return _make_response(
        category="Compound infrastructure degradation",
        severity="high",
        severity_reason=(
            "Simultaneous DB saturation and SMS gateway failure affect payments and alerts."
        ),
        summary="Concurrent DB CPU saturation slowing payments and SMS gateway timeouts.",
        hypothesis_title="DB CPU saturation from reporting queries",
        hypothesis_reasoning="Reporting load on primary DB correlates with payment slowdown.",
        next_steps=[
            "Kill long-running reporting queries.",
            "Redirect reporting to read replica.",
            "Monitor payment latency after change.",
        ],
        extra_hypotheses=[{
            "title": "External SMS gateway connectivity failure",
            "reasoning": "SMS timeouts logged independently of DB issue.",
            "next_steps": [
                "Check SMS provider status page.",
                "Verify outbound network routes to SMS API.",
            ],
        }],
        affected_users="Customers transacting and awaiting SMS codes",
    )


@pytest.fixture
def eval_network_routing_response() -> str:
    return _make_response(
        category="Network routing issue",
        severity="medium",
        severity_reason=(
            "Intermittent routing failures degrade a subset of services "
            "but core transactions still complete for most users."
        ),
        summary="Intermittent packet loss on internal service mesh causing sporadic failures.",
        hypothesis_title="BGP route flap or misconfigured network policy",
        hypothesis_reasoning=(
            "Packet loss affects multiple services simultaneously, "
            "suggesting a shared network layer issue rather than individual service bugs."
        ),
        next_steps=[
            "Check network switch and router logs for BGP events.",
            "Inspect service mesh (Istio/Linkerd) for recent policy changes.",
            "Run traceroute between affected service pods.",
        ],
        affected_users="Subset of users on affected network segments",
    )
