"""
Typed domain exceptions for the agent layer.

The hierarchy is designed so that callers can catch at the right granularity:

  LLMAuthenticationError          -- permanent config error (wrong API key);
                                     maps to HTTP 500 (server misconfiguration).

  LLMUnavailableError             -- transient service error (connection,
                                     timeout, unexpected API failure);
                                     maps to HTTP 503.

    LLMRateLimitError             -- sub-class of LLMUnavailableError;
                                     the API rate limit was exceeded.
                                     Retried automatically with exponential
                                     backoff by IncidentAnalyzer; if all
                                     attempts fail it propagates as 503.

  LLMInvalidResponseError         -- the LLM failed to produce a schema-valid
                                     response after all validation retries;
                                     maps to HTTP 422.
"""


class LLMAuthenticationError(Exception):
    """
    Raised when the LLM API rejects the request due to invalid credentials.

    This is a permanent, non-retryable error that indicates a server-side
    configuration problem (e.g. OPENAI_API_KEY is wrong or revoked).
    """


class LLMUnavailableError(Exception):
    """
    Raised when the LLM API is unreachable or returns a transient error.

    Covers network failures, unexpected HTTP errors, and empty responses.
    Retrying after a delay may resolve the issue.
    """


class LLMRateLimitError(LLMUnavailableError):
    """
    Raised when the API rate limit or quota has been exceeded.

    Inherits from LLMUnavailableError so that callers catching the base class
    still handle rate-limit failures correctly.  IncidentAnalyzer retries
    these automatically with exponential backoff before re-raising.
    """


class LLMInvalidResponseError(Exception):
    """
    Raised when the LLM fails to produce a schema-valid response after all
    validation retry attempts have been exhausted.
    """
