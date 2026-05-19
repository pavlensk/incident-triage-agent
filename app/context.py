SYSTEM_ARCHITECTURE = """
Payment Platform Services:
- api-gateway: receives external HTTP requests from clients and routes them to internal services.
- auth-service: authentication and token issuance.
- payment-service: creation and processing of payment transactions. Uses a dedicated PostgreSQL instance.
- billing-service: balance management and invoicing. Uses a dedicated PostgreSQL instance.
- notification-service: sending e-mail and SMS notifications.
- reporting-service: generating reports and exporting data. Extra load on DB with long analytical queries.

General notes: 
All services write logs to a centralized log storage (ELK).
Payments often experience external provider errors (timeout, 5xx, invalid credentials).
notification-service may degrade when external SMTP/SMS providers have issues.
"""

PAST_INCIDENTS_LIST = [
    "[INC-101] Customers report they cannot pay by card. payment-service logs show massive timeouts calling PayGate provider. Issue started around 12:05 UTC. No other metric anomalies.",
    "[INC-102] Sharp increase in response time for /payments/create (up to 5-7 seconds). DB dashboards show high CPU and many long-running queries from reporting-service. Some customers receive 504 Gateway Timeout.",
    "[INC-103] Users do not receive top-up confirmation e-mails. Money credited successfully. notification-service logs show intermittent connection errors to SMTP provider.",
    "[INC-104] Some customers cannot log in via mobile app. auth-service responds with 401 errors. Logs show invalid token signatures. No other failures."
]