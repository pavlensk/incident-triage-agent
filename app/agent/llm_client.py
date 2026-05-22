"""
LLM client abstraction (Dependency Inversion Principle).

LLMClientProtocol is a structural protocol (typing.Protocol) that:
  - allows injecting deterministic stubs in tests without monkey-patching
    -- any object implementing complete() satisfies the protocol;
  - makes swapping OpenAI for Anthropic, Gemini, or a local model trivial.

OpenAILLMClient is the concrete production implementation on top of
openai.AsyncOpenAI.  It translates SDK-specific exceptions into the typed
domain exceptions defined in exceptions.py so that the rest of the pipeline
never depends on openai internals:

  openai.AuthenticationError          -> LLMAuthenticationError  (permanent)
  openai.RateLimitError               -> LLMRateLimitError        (retryable)
  openai.APIConnectionError /
  openai.APITimeoutError              -> LLMUnavailableError      (transient)
  any other Exception                 -> LLMUnavailableError      (catch-all)
  None / empty response content       -> LLMUnavailableError
"""
import logging
from typing import List, Dict, Protocol, runtime_checkable

import openai
from openai import AsyncOpenAI

from app.agent.exceptions import (
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMUnavailableError,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class LLMClientProtocol(Protocol):
    """Minimal interface that any LLM client must satisfy."""

    async def complete(self, messages: List[Dict[str, str]]) -> str:
        """Send a list of messages and return the model's response text."""
        ...


class OpenAILLMClient:
    """
    LLMClientProtocol implementation on top of the OpenAI Chat Completions API.

    Parameters
    ----------
    api_key :
        OpenAI API key.  Passed explicitly -- no os.getenv() inside this class.
    model_name :
        Chat model identifier (e.g. ``"gpt-4o-mini"``).
    temperature :
        Sampling temperature (0.0 – 2.0).
    timeout :
        Per-request HTTP timeout in seconds.  Prevents hung requests from
        blocking the event loop indefinitely.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str,
        temperature: float,
        timeout: float = 30.0,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, timeout=timeout)
        self._model_name = model_name
        self._temperature = temperature

    async def complete(self, messages: List[Dict[str, str]]) -> str:
        """
        Call Chat Completions and return the response text.

        Raises
        ------
        LLMAuthenticationError
            If the API key is invalid or revoked (permanent error).
        LLMRateLimitError
            If the rate limit or quota has been exceeded (retryable).
        LLMUnavailableError
            For any other transient failure (connection, timeout, empty
            response, or unexpected SDK error).
        """
        try:
            response = await self._client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=self._temperature,
            )
            content = response.choices[0].message.content
            if not content:
                # The API returned a well-formed response but with an empty
                # or null content field -- treat as a transient service error.
                raise LLMUnavailableError(
                    "LLM returned an empty response content."
                )
            return content

        except (LLMUnavailableError, LLMAuthenticationError, LLMRateLimitError):
            # Re-raise our own exceptions unchanged (e.g. the empty-content
            # check above) so they are not swallowed by the bare handler below.
            raise

        except openai.AuthenticationError as exc:
            logger.error("OpenAI authentication failure: %s", exc)
            raise LLMAuthenticationError(
                f"OpenAI authentication failed -- check OPENAI_API_KEY: {exc}"
            ) from exc

        except openai.RateLimitError as exc:
            logger.warning("OpenAI rate limit exceeded: %s", exc)
            raise LLMRateLimitError(
                f"OpenAI rate limit exceeded: {exc}"
            ) from exc

        except (openai.APIConnectionError, openai.APITimeoutError) as exc:
            logger.error("OpenAI connection/timeout error: %s", exc)
            raise LLMUnavailableError(
                f"OpenAI connection error: {exc}"
            ) from exc

        except Exception as exc:
            logger.error("Unexpected OpenAI API error: %s", exc)
            raise LLMUnavailableError(
                f"Unexpected OpenAI error: {exc}"
            ) from exc
