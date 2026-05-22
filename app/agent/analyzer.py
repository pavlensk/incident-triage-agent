"""
IncidentAnalyzer -- pipeline orchestrator for incident analysis.

Single responsibility: coordinate four stages:
  1. Input text parsing                (InputParser.parse_input)
  2. Relevant context retrieval        (ContextRetriever.retrieve)
  3. Prompt assembly                   (PromptBuilder.build_system_prompt)
  4. LLM call + self-correction loop   (LLMClientProtocol + Pydantic)

Two independent retry mechanisms protect against different failure modes:

  _call_llm()   -- retries on LLMRateLimitError with exponential backoff.
                   LLMUnavailableError and LLMAuthenticationError propagate
                   immediately (no backoff -- connection failures are unlikely
                   to self-heal within a single request cycle).

  analyze()     -- retries on schema ValidationError (self-correction loop):
                   appends the error details to the conversation so the LLM
                   can fix its own output.

Dependencies are injected via the constructor (Dependency Inversion Principle),
making this component fully testable without a real API key.
"""
import asyncio
import logging

from pydantic import ValidationError

from app.schemas import IncidentAnalysis
from app.agent.llm_client import LLMClientProtocol
from app.agent.input_parser import InputParser
from app.agent.retriever import ContextRetriever
from app.agent.prompt_builder import PromptBuilder
from app.agent.exceptions import (
    LLMInvalidResponseError,
    LLMRateLimitError,
    LLMUnavailableError,
)

logger = logging.getLogger(__name__)


class IncidentAnalyzer:
    """
    Orchestrator for the incident analysis agent.

    Parameters
    ----------
    llm_client :
        Any object satisfying LLMClientProtocol.
        Use MockLLMClient in tests, OpenAILLMClient in production.
    input_parser :
        Component responsible for input normalisation and keyword extraction.
    retriever :
        Component responsible for context retrieval.
    prompt_builder :
        Component responsible for prompt assembly.
    max_retries :
        Maximum attempts to obtain a schema-valid response (validation loop).
    llm_retry_attempts :
        Maximum attempts on transient rate-limit errors before giving up.
    llm_retry_delay_seconds :
        Base delay (seconds) for exponential backoff on rate-limit retries.
        Set to 0.0 in tests to keep the suite fast.
    """

    def __init__(
        self,
        llm_client: LLMClientProtocol,
        input_parser: InputParser,
        retriever: ContextRetriever,
        prompt_builder: PromptBuilder,
        max_retries: int = 3,
        llm_retry_attempts: int = 2,
        llm_retry_delay_seconds: float = 1.0,
    ) -> None:
        self._llm = llm_client
        self._input_parser = input_parser
        self._retriever = retriever
        self._prompt_builder = prompt_builder
        self._max_retries = max_retries
        self._llm_retry_attempts = llm_retry_attempts
        self._llm_retry_delay = llm_retry_delay_seconds

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _call_llm(self, messages: list) -> str:
        """
        Call the LLM with automatic retry on rate-limit errors.

        Retries up to ``llm_retry_attempts`` times with exponential backoff
        when a LLMRateLimitError is received.  All other exceptions propagate
        immediately -- there is no point retrying authentication failures or
        generic connection errors within a single request cycle.
        """
        for attempt in range(1, self._llm_retry_attempts + 1):
            try:
                return await self._llm.complete(messages)
            except LLMRateLimitError:
                if attempt == self._llm_retry_attempts:
                    logger.error(
                        "Rate limit hit on final attempt (%d/%d); giving up.",
                        attempt, self._llm_retry_attempts,
                    )
                    raise
                delay = self._llm_retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    "Rate limit hit (attempt %d/%d); retrying in %.1f s.",
                    attempt, self._llm_retry_attempts, delay,
                )
                await asyncio.sleep(delay)

        # Unreachable -- satisfies static type checkers.
        raise LLMUnavailableError("LLM retry loop exited without a result.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze(self, user_input: str) -> IncidentAnalysis:
        """
        Run the full incident analysis pipeline.

        Returns a validated IncidentAnalysis instance.

        Raises
        ------
        LLMAuthenticationError
            Propagated from the LLM client if the API key is invalid.
        LLMUnavailableError / LLMRateLimitError
            Propagated when the LLM is unreachable and retries are exhausted.
        LLMInvalidResponseError
            When schema validation fails on every attempt of the correction loop.
        """
        # Stage 1: parse input text
        parsed = self._input_parser.parse_input(user_input)

        # Stage 2: retrieve relevant past incidents
        context = self._retriever.retrieve(parsed)

        # Stage 3: assemble the system prompt
        system_prompt = self._prompt_builder.build_system_prompt(context)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analyze this incident:\n{parsed['raw_text']}"},
        ]

        # Stage 4: call the LLM with a self-correction loop on validation errors
        for attempt in range(1, self._max_retries + 1):
            # _call_llm handles rate-limit retries internally;
            # other LLM exceptions propagate to the caller.
            raw_response = await self._call_llm(messages)

            try:
                result = IncidentAnalysis.model_validate_json(raw_response)
                logger.info("Analysis completed successfully (attempt %d).", attempt)
                return result

            except ValidationError as exc:
                logger.warning(
                    "Attempt %d/%d: schema validation failed -- %s",
                    attempt, self._max_retries, exc.errors(),
                )
                if attempt == self._max_retries:
                    raise LLMInvalidResponseError(
                        f"Failed to generate a valid response after {self._max_retries} attempts."
                    ) from exc

                # Append the invalid response and correction instructions to
                # the conversation so the LLM can fix its own output.
                messages.append({"role": "assistant", "content": raw_response})
                messages.append({
                    "role": "user",
                    "content": self._prompt_builder.build_correction_message(exc.json()),
                })

        # Unreachable -- satisfies static type checkers.
        raise LLMInvalidResponseError("Analysis pipeline exited without a result.")
