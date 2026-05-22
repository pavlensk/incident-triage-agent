"""
Unit tests for OpenAILLMClient -- exception mapping, content guards, passthrough.

All tests are fully offline: AsyncOpenAI is patched at the module boundary so no
HTTP connection is ever attempted.  Each test class targets a concrete branch in
complete() that is invisible to the existing MockLLMClient-based suite:

  TestOpenAIExceptionMapping  -- all five except-branches (SDK -> domain exc)
  TestResponseContentGuard    -- if-not-content guard (None / "" / valid)
  TestOwnExceptionPassthrough -- re-raise guard prevents double-wrapping
"""
import openai
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.agent.llm_client import OpenAILLMClient
from app.agent.exceptions import (
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMUnavailableError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_api_status_error(exc_class, status_code: int):
    """
    Construct a minimal openai APIStatusError subclass instance.

    APIStatusError.__init__ reads response.request and response.status_code.
    A plain MagicMock satisfies both attribute reads without requiring a real
    httpx.Response, keeping the helper infrastructure-free.
    """
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.request = MagicMock()
    return exc_class(message="simulated API error", response=mock_response, body=None)


def _make_connection_error() -> openai.APIConnectionError:
    """Minimal APIConnectionError -- only a request object is required."""
    return openai.APIConnectionError(request=MagicMock())


def _make_timeout_error() -> openai.APITimeoutError:
    """Minimal APITimeoutError -- subclass of APIConnectionError."""
    return openai.APITimeoutError(request=MagicMock())


def _make_openai_response(content: str | None) -> MagicMock:
    """
    Return a mock that mirrors response.choices[0].message.content.

    choices is an explicit list so the mock accurately reflects the SDK's
    structure -- mock.choices[0] via __getitem__ on an auto-created MagicMock
    would work today but wouldn't express intent and could silently diverge
    from the real SDK shape if the attribute access pattern ever changed.
    """
    choice = MagicMock()
    choice.message.content = content
    mock = MagicMock()
    mock.choices = [choice]
    return mock


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def llm_and_mock():
    """
    Yield (OpenAILLMClient, mock_create) with AsyncOpenAI replaced by a MagicMock.

    The patch is applied at the module level so the constructor receives the
    mock class when it calls AsyncOpenAI(...).  The context manager remains
    active for the full duration of the test, then restores the real class.
    """
    with patch("app.agent.llm_client.AsyncOpenAI") as MockOpenAI:
        mock_openai_instance = MagicMock()
        MockOpenAI.return_value = mock_openai_instance

        mock_create = AsyncMock()
        mock_openai_instance.chat.completions.create = mock_create

        client = OpenAILLMClient(
            api_key="test-key",
            model_name="gpt-4o-mini",
            temperature=0.1,
            timeout=5.0,
        )
        yield client, mock_create


# ---------------------------------------------------------------------------
# Class 1: SDK exception -> domain exception mapping (all five branches)
# ---------------------------------------------------------------------------

class TestOpenAIExceptionMapping:
    """
    Verify that every branch in the except-ladder of complete() fires correctly.

    Each test raises a specific SDK exception via mock side_effect, then asserts:
      1. The correct domain exception type propagates to the caller.
      2. The original SDK exception is chained as __cause__ for traceability.

    Getting these mappings wrong has silent, high-impact consequences:
      - A misrouted RateLimitError means the analyzer never triggers its retry loop.
      - A swallowed AuthenticationError surfaces as an opaque 503 instead of 500.
    """

    async def test_authentication_error_maps_to_llm_auth_error(self, llm_and_mock):
        """
        openai.AuthenticationError (401) -> LLMAuthenticationError.

        LLMAuthenticationError is the only non-retryable domain exception.  If
        it were mapped to LLMUnavailableError instead, the analyzer would silently
        retry a permanently broken API key until max_attempts are exhausted, then
        return HTTP 503 -- obscuring a configuration problem as a transient one.
        """
        client, mock_create = llm_and_mock
        sdk_exc = _make_api_status_error(openai.AuthenticationError, 401)
        mock_create.side_effect = sdk_exc

        with pytest.raises(LLMAuthenticationError) as exc_info:
            await client.complete([{"role": "user", "content": "test incident"}])

        assert exc_info.value.__cause__ is sdk_exc

    async def test_rate_limit_error_maps_to_llm_rate_limit_error(self, llm_and_mock):
        """
        openai.RateLimitError (429) -> LLMRateLimitError.

        LLMRateLimitError is the single exception IncidentAnalyzer retries with
        exponential backoff.  Mapping it to plain LLMUnavailableError disables
        the retry loop with no visible failure -- a silent regression that only
        appears under production load when the rate limit is genuinely hit.
        """
        client, mock_create = llm_and_mock
        sdk_exc = _make_api_status_error(openai.RateLimitError, 429)
        mock_create.side_effect = sdk_exc

        with pytest.raises(LLMRateLimitError) as exc_info:
            await client.complete([{"role": "user", "content": "test incident"}])

        assert exc_info.value.__cause__ is sdk_exc
        # LLMRateLimitError IS-A LLMUnavailableError -- the global HTTP 503 handler
        # catches both via the base class, so no additional handler is needed.
        assert isinstance(exc_info.value, LLMUnavailableError)

    @pytest.mark.parametrize("sdk_exc_factory,label", [
        (_make_connection_error, "APIConnectionError"),
        (_make_timeout_error,    "APITimeoutError"),
    ])
    async def test_transport_errors_map_to_unavailable(
        self, llm_and_mock, sdk_exc_factory, label
    ):
        """
        openai.APIConnectionError / APITimeoutError -> LLMUnavailableError.

        Both are transient transport failures.  They must NOT map to
        LLMRateLimitError -- that would incorrectly trigger exponential backoff
        for plain connection drops, adding latency with no benefit.
        """
        client, mock_create = llm_and_mock
        sdk_exc = sdk_exc_factory()
        mock_create.side_effect = sdk_exc

        with pytest.raises(LLMUnavailableError) as exc_info:
            await client.complete([{"role": "user", "content": "test incident"}])

        assert exc_info.value.__cause__ is sdk_exc
        assert not isinstance(exc_info.value, LLMRateLimitError), (
            f"{label} must map to base LLMUnavailableError, not LLMRateLimitError"
        )

    async def test_unexpected_exception_maps_to_unavailable_via_catchall(
        self, llm_and_mock
    ):
        """
        Any unrecognised Exception -> LLMUnavailableError (bare except branch).

        This is the safety net for future openai SDK versions that introduce
        exception types not yet explicitly handled.  Without it, an unknown
        exception would escape as an unhandled 500 with a raw traceback visible
        in the API response body.
        """
        client, mock_create = llm_and_mock
        sdk_exc = RuntimeError("Unexpected internal SDK failure")
        mock_create.side_effect = sdk_exc

        with pytest.raises(LLMUnavailableError) as exc_info:
            await client.complete([{"role": "user", "content": "test incident"}])

        assert exc_info.value.__cause__ is sdk_exc


# ---------------------------------------------------------------------------
# Class 2: None / empty content guard
# ---------------------------------------------------------------------------

class TestResponseContentGuard:
    """
    The OpenAI API can return HTTP 200 with a null or empty content field when
    the model output is suppressed by the content-moderation layer.  Without the
    guard at lines 95-100, the call to 'return content' would propagate None to
    the Pydantic validation step, where it would fail with a confusing TypeError
    rather than the expected LLMUnavailableError -> HTTP 503 path.
    """

    @pytest.mark.parametrize("content,label", [
        (None, "None (content-filtered response)"),
        ("",   "empty string (zero-length body)"),
    ])
    async def test_falsy_content_raises_unavailable_with_clear_message(
        self, llm_and_mock, content, label
    ):
        """
        Both None and '' are falsy and must trigger the guard identically.

        The error message must contain 'empty' so operators can distinguish
        this failure mode from a transport error when scanning logs.
        __cause__ must be None because this guard raises a first-party exception
        -- it does not wrap a third-party one.
        """
        client, mock_create = llm_and_mock
        mock_create.return_value = _make_openai_response(content)

        with pytest.raises(LLMUnavailableError, match="empty") as exc_info:
            await client.complete([{"role": "user", "content": "analyze this incident"}])

        assert exc_info.value.__cause__ is None, (
            f"Content guard ({label}) must raise a fresh LLMUnavailableError "
            "with no __cause__, not a wrapped third-party exception."
        )

    async def test_valid_content_is_returned_unchanged(self, llm_and_mock):
        """
        A non-empty response string passes through the guard and is returned as-is.

        mock_create.assert_awaited_once() confirms the underlying API call was
        made exactly once -- i.e. the mock was not bypassed and the happy-path
        branch (lines 87-101) executed fully.
        """
        client, mock_create = llm_and_mock
        expected = '{"category": "External payment provider issue", "severity": "high"}'
        mock_create.return_value = _make_openai_response(expected)

        result = await client.complete([{"role": "user", "content": "analyze this incident"}])

        assert result == expected
        mock_create.assert_awaited_once()


# ---------------------------------------------------------------------------
# Class 3: Own-exception re-raise passthrough
# ---------------------------------------------------------------------------

class TestOwnExceptionPassthrough:
    """
    Lines 103-106 re-raise domain exceptions without re-wrapping them.

    The guard exists because a LLMUnavailableError raised inside the try block
    (by the content guard at lines 95-100) would otherwise fall through to the
    bare 'except Exception' at line 126 and be wrapped a second time:

        LLMUnavailableError("Unexpected OpenAI error: LLM returned an empty ...")

    instead of the original:

        LLMUnavailableError("LLM returned an empty response content.")

    Double-wrapping corrupts the error message, inflates the exception chain,
    and makes log-based incident triage harder.
    """

    async def test_content_guard_exception_is_not_double_wrapped(self, llm_and_mock):
        """
        LLMUnavailableError from the content guard must exit with its message
        intact.  'Unexpected OpenAI error' in the message is the observable
        fingerprint of double-wrapping by the catch-all handler.
        """
        client, mock_create = llm_and_mock
        mock_create.return_value = _make_openai_response(None)

        with pytest.raises(LLMUnavailableError) as exc_info:
            await client.complete([{"role": "user", "content": "test incident"}])

        message = str(exc_info.value)
        assert "Unexpected OpenAI error" not in message, (
            "LLMUnavailableError from the content guard was re-wrapped by the "
            "catch-all handler -- the re-raise guard at lines 103-106 is broken."
        )
        assert "empty" in message.lower(), (
            "The original 'empty response content' message must survive the "
            "re-raise guard unchanged."
        )


# ---------------------------------------------------------------------------
# Class 4: base_url forwarding
# ---------------------------------------------------------------------------

class TestBaseUrlForwarding:
    """
    OpenAILLMClient must forward the base_url parameter to AsyncOpenAI so that
    corporate proxies, Azure OpenAI endpoints, and local model servers
    (LiteLLM, Ollama) work without any changes to the client code.

    Two cases are verified:
      - base_url=None  -> AsyncOpenAI is called WITHOUT a base_url kwarg
                          (the SDK uses its own default: api.openai.com).
      - base_url=<url> -> AsyncOpenAI is called WITH the exact url supplied.

    Passing None explicitly would override the SDK default in some versions,
    so the implementation uses conditional kwargs unpacking.
    """

    def test_base_url_is_forwarded_when_provided(self):
        """
        When base_url is supplied, AsyncOpenAI must receive it as a keyword
        argument.  This is the critical path for proxy / Azure deployments.
        """
        proxy_url = "https://my-corp-proxy.example.com/openai/v1"
        with patch("app.agent.llm_client.AsyncOpenAI") as MockOpenAI:
            MockOpenAI.return_value = MagicMock()
            OpenAILLMClient(
                api_key="test-key",
                model_name="gpt-4o-mini",
                temperature=0.0,
                base_url=proxy_url,
            )
        _, kwargs = MockOpenAI.call_args
        assert kwargs.get("base_url") == proxy_url, (
            f"AsyncOpenAI was not called with base_url={proxy_url!r}. "
            f"Actual kwargs: {kwargs}"
        )

    def test_base_url_is_absent_when_not_provided(self):
        """
        When base_url is None (the default), AsyncOpenAI must NOT receive a
        base_url kwarg at all -- passing None explicitly could override the
        SDK's built-in default in some SDK versions.
        """
        with patch("app.agent.llm_client.AsyncOpenAI") as MockOpenAI:
            MockOpenAI.return_value = MagicMock()
            OpenAILLMClient(
                api_key="test-key",
                model_name="gpt-4o-mini",
                temperature=0.0,
                # base_url intentionally omitted (defaults to None)
            )
        _, kwargs = MockOpenAI.call_args
        assert "base_url" not in kwargs, (
            "AsyncOpenAI must not receive base_url when it was not specified -- "
            f"got base_url={kwargs.get('base_url')!r} in kwargs."
        )
