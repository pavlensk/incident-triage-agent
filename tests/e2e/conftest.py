"""
E2E test fixtures: gold-standard LLM responses for the five canonical
incident types from the assignment.

Each fixture represents a realistic, schema-valid LLM response for a specific
scenario.  E2E tests inject these via MockLLMClient so the full HTTP → pipeline
→ response path is exercised with realistic data, not synthetic stubs.
"""
import json
import pytest


@pytest.fixture
def paygate_llm_response() -> str:
    """Gold-standard response: external payment provider failure (INC-101 type)."""
    return json.dumps({
        "category": "External payment provider issue",
        "severity": "high",
        "severity_reason": (
            "Core financial operations are fully blocked: customers cannot complete "
            "card payments, directly impacting revenue."
        ),
        "affected_users": "All customers attempting card payments",
        "summary": (
            "PayGate is returning connection timeouts starting at 12:05 UTC, "
            "blocking all outbound card payment processing."
        ),
        "hypotheses": [
            {
                "title": "Degradation or outage on the PayGate side",
                "reasoning": (
                    "Timeouts are observed exclusively when calling PayGate; "
                    "all internal services remain stable, ruling out internal causes."
                ),
                "next_steps": [
                    "Check the PayGate public status page and provider incident feed.",
                    "Compare latency and error-rate metrics across all payment providers.",
                    "Temporarily route traffic to a backup payment gateway.",
                ],
            }
        ],
    })


@pytest.fixture
def db_saturation_llm_response() -> str:
    """Gold-standard response: database resource saturation (INC-102 type)."""
    return json.dumps({
        "category": "Database resource saturation",
        "severity": "high",
        "severity_reason": (
            "Payment creation is timing out due to DB CPU exhaustion, "
            "causing cascading 504 errors from the API gateway."
        ),
        "affected_users": "All users creating new payment transactions",
        "summary": (
            "reporting-service analytical queries are saturating PostgreSQL CPU, "
            "blocking payment-service DB connections and causing cascade timeouts."
        ),
        "hypotheses": [
            {
                "title": "reporting-service queries exhausting PostgreSQL CPU",
                "reasoning": (
                    "DB dashboards show CPU spike correlated with payment slowdown. "
                    "Long-running analytical queries from reporting-service are competing "
                    "with transactional workloads on the same PostgreSQL instance."
                ),
                "next_steps": [
                    "Run pg_stat_activity to identify and kill long-running queries.",
                    "Check connection pool saturation on the payment-service DB instance.",
                    "Restrict reporting-service queries to a read replica immediately.",
                ],
            }
        ],
    })


@pytest.fixture
def auth_failure_llm_response() -> str:
    """Gold-standard response: authentication token failure (INC-104 type)."""
    return json.dumps({
        "category": "Authentication token failure",
        "severity": "high",
        "severity_reason": (
            "Authentication is fully blocked for mobile users; "
            "no user can log in or refresh a session."
        ),
        "affected_users": "All mobile app users attempting to authenticate",
        "summary": (
            "auth-service is rejecting tokens with invalid signature errors across "
            "all pods, completely blocking mobile login flows."
        ),
        "hypotheses": [
            {
                "title": "JWT signing key mismatch across auth-service pods",
                "reasoning": (
                    "Invalid token signatures across all pods suggest the signing key "
                    "used at token creation differs from the verification key, "
                    "pointing to a recent secret rotation or deployment."
                ),
                "next_steps": [
                    "Check recent secret rotation events in Vault or the config system.",
                    "Verify JWT public key consistency across all auth-service pod replicas.",
                    "Inspect auth-service logs for key-loading or cryptographic errors.",
                ],
            }
        ],
    })


@pytest.fixture
def smtp_degradation_llm_response() -> str:
    """Gold-standard response: notification delivery degradation (INC-103 type)."""
    return json.dumps({
        "category": "Notification delivery degradation",
        "severity": "low",
        "severity_reason": (
            "Financial operations are fully intact; only email confirmation "
            "delivery is delayed -- a non-critical user-facing side effect."
        ),
        "affected_users": "Users expecting top-up confirmation emails",
        "summary": (
            "notification-service is failing to connect to the external SMTP provider, "
            "causing confirmation emails to be delayed or dropped."
        ),
        "hypotheses": [
            {
                "title": "External SMTP provider outage or network routing issue",
                "reasoning": (
                    "Intermittent connection timeouts to SMTP are logged while "
                    "billing records confirm money was credited correctly, "
                    "isolating the failure to the notification delivery path."
                ),
                "next_steps": [
                    "Check SMTP provider status page and recent maintenance windows.",
                    "Inspect the notification-service retry queue and dead-letter metrics.",
                    "Verify outbound network routing from the cluster to the SMTP endpoint.",
                ],
            }
        ],
    })


@pytest.fixture
def compound_degradation_llm_response() -> str:
    """Gold-standard response: compound multi-service degradation."""
    return json.dumps({
        "category": "Compound infrastructure degradation",
        "severity": "high",
        "severity_reason": (
            "Simultaneous degradation of payment processing and SMS notification "
            "delivery affects both core transactions and user communication."
        ),
        "affected_users": "Customers creating payments and users expecting SMS alerts",
        "summary": (
            "The platform has two concurrent failures: DB CPU saturation slowing "
            "payments, and external SMS gateway timeouts blocking notifications."
        ),
        "hypotheses": [
            {
                "title": "DB CPU saturation from reporting queries slowing payments",
                "reasoning": (
                    "reporting-service load on the primary DB correlates with payment "
                    "slowness; isolated from the SMS issue which has a separate root cause."
                ),
                "next_steps": [
                    "Kill long-running reporting queries on pg_stat_activity.",
                    "Redirect reporting-service to a read replica to free primary DB.",
                    "Monitor payment latency after reporting load is reduced.",
                ],
            },
            {
                "title": "External SMS gateway connectivity failure",
                "reasoning": (
                    "Network timeout errors to the SMS gateway are logged independently "
                    "of the DB issue, suggesting an external provider problem."
                ),
                "next_steps": [
                    "Check the SMS gateway provider status page.",
                    "Verify outbound network routes to the SMS API from the cluster.",
                ],
            },
        ],
    })
