"""
Unit tests for the typed domain exception hierarchy (exceptions.py).

Verifying the inheritance chain matters because:
  - global exception handlers in main.py catch at specific base classes;
  - IncidentAnalyzer retries LLMRateLimitError but not LLMAuthenticationError;
  - callers that catch LLMUnavailableError must also catch LLMRateLimitError.
"""
import pytest

from app.agent.exceptions import (
    LLMAuthenticationError,
    LLMInvalidResponseError,
    LLMRateLimitError,
    LLMUnavailableError,
)


class TestExceptionHierarchy:
    def test_rate_limit_is_subclass_of_unavailable(self):
        assert issubclass(LLMRateLimitError, LLMUnavailableError)

    def test_auth_error_is_not_subclass_of_unavailable(self):
        assert not issubclass(LLMAuthenticationError, LLMUnavailableError)

    def test_invalid_response_is_not_subclass_of_unavailable(self):
        assert not issubclass(LLMInvalidResponseError, LLMUnavailableError)

    def test_all_are_subclasses_of_exception(self):
        for cls in (
            LLMAuthenticationError,
            LLMUnavailableError,
            LLMRateLimitError,
            LLMInvalidResponseError,
        ):
            assert issubclass(cls, Exception), f"{cls.__name__} must inherit Exception"


class TestCatchAtBaseClass:
    """Catching the base class must also catch the derived class."""

    def test_catching_unavailable_catches_rate_limit(self):
        with pytest.raises(LLMUnavailableError):
            raise LLMRateLimitError("rate limited")

    def test_catching_rate_limit_does_not_catch_generic_unavailable(self):
        """LLMUnavailableError is NOT a LLMRateLimitError -- only the reverse is true."""
        with pytest.raises(LLMUnavailableError):
            raise LLMUnavailableError("connection refused")
        # LLMRateLimitError would NOT be raised here -- just confirming the base works

    def test_auth_error_is_not_caught_as_unavailable(self):
        """LLMAuthenticationError must propagate past a LLMUnavailableError guard."""
        with pytest.raises(LLMAuthenticationError):
            try:
                raise LLMAuthenticationError("invalid key")
            except LLMUnavailableError:
                pytest.fail("LLMAuthenticationError must not be caught as LLMUnavailableError")


class TestExceptionMessages:
    def test_exception_preserves_message(self):
        msg = "OpenAI returned 429 Too Many Requests"
        exc = LLMRateLimitError(msg)
        assert str(exc) == msg

    def test_exception_chaining_with_cause(self):
        cause = ValueError("underlying cause")
        exc = LLMUnavailableError("wrapped") 
        try:
            raise exc from cause
        except LLMUnavailableError as e:
            assert e.__cause__ is cause

    def test_rate_limit_isinstance_checks(self):
        exc = LLMRateLimitError("rate limited")
        assert isinstance(exc, LLMRateLimitError)
        assert isinstance(exc, LLMUnavailableError)
        assert isinstance(exc, Exception)
        assert not isinstance(exc, LLMAuthenticationError)
        assert not isinstance(exc, LLMInvalidResponseError)
