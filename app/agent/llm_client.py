"""
LLM client abstraction (Dependency Inversion Principle).

LLMClientProtocol is a structural protocol (typing.Protocol) that:
  - allows injecting deterministic stubs in tests without monkey-patching
    -- any object implementing complete() satisfies the protocol;
  - makes swapping OpenAI for Anthropic, Gemini, or a local model trivial.

OpenAILLMClient is the concrete production implementation on top of openai.AsyncOpenAI.
"""
import logging
from typing import List, Dict, Protocol, runtime_checkable

from openai import AsyncOpenAI

from app.agent.exceptions import LLMUnavailableError

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

    Accepts configuration parameters explicitly (no direct os.getenv calls),
    which simplifies testing and follows the principle of explicit dependencies.
    """

    def __init__(self, api_key: str, model_name: str, temperature: float) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model_name = model_name
        self._temperature = temperature

    async def complete(self, messages: List[Dict[str, str]]) -> str:
        """
        Call Chat Completions and return the response text.

        All network / API errors are wrapped in LLMUnavailableError so that
        upstream code does not depend on openai SDK internals.
        """
        try:
            response = await self._client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=self._temperature,
            )
            return response.choices[0].message.content
        except Exception as exc:
            logger.error("OpenAI API error: %s", exc)
            raise LLMUnavailableError(f"OpenAI API unavailable: {exc}") from exc
