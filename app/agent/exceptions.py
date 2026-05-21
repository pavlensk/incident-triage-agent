"""
Typed domain exceptions for the agent layer.

Separating by error type lets main.py return precise HTTP status codes
(503 vs 422) without catching a bare Exception.
"""


class LLMUnavailableError(Exception):
    """Raised when the LLM API is unreachable or returns a network/transport error."""


class LLMInvalidResponseError(Exception):
    """Raised when the LLM fails to produce a schema-valid response after all retry attempts."""
