"""
IncidentAnalyzer -- pipeline orchestrator for incident analysis.

Single responsibility: coordinate four stages:
  1. Input text parsing                (ContextRetriever.parse_input)
  2. Relevant context retrieval        (ContextRetriever.retrieve)
  3. Prompt assembly                   (PromptBuilder.build_system_prompt)
  4. LLM call + self-correction loop   (LLMClientProtocol + Pydantic)

Dependencies are injected via the constructor (Dependency Inversion Principle),
making this component fully testable without a real API key.
"""
import logging
from typing import Dict, Any

from pydantic import ValidationError

from app.schemas import IncidentAnalysis
from app.agent.llm_client import LLMClientProtocol
from app.agent.retriever import ContextRetriever
from app.agent.prompt_builder import PromptBuilder
from app.agent.exceptions import LLMInvalidResponseError

logger = logging.getLogger(__name__)


class IncidentAnalyzer:
    """
    Orchestrator for the incident analysis agent.

    Parameters
    ----------
    llm_client :
        Any object satisfying LLMClientProtocol.
        Use MockLLMClient in tests, OpenAILLMClient in production.
    retriever :
        Component responsible for input parsing and context retrieval.
    prompt_builder :
        Component responsible for prompt assembly.
    max_retries :
        Maximum number of attempts to obtain a schema-valid response from the LLM.
    """

    def __init__(
        self,
        llm_client: LLMClientProtocol,
        retriever: ContextRetriever,
        prompt_builder: PromptBuilder,
        max_retries: int = 3,
    ) -> None:
        self._llm = llm_client
        self._retriever = retriever
        self._prompt_builder = prompt_builder
        self._max_retries = max_retries

    async def analyze(self, user_input: str) -> Dict[str, Any]:
        """
        Run the full incident analysis pipeline.

        Returns a dict conforming to the IncidentAnalysis schema.
        Raises LLMInvalidResponseError when all retry attempts are exhausted.
        LLMUnavailableError propagates upward without being caught here.
        """
        # Stage 1: parse input text
        parsed = self._retriever.parse_input(user_input)

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
            raw_response = await self._llm.complete(messages)  # may raise LLMUnavailableError

            try:
                result = IncidentAnalysis.model_validate_json(raw_response)
                logger.info("Analysis completed successfully (attempt %d).", attempt)
                return result.model_dump()

            except ValidationError as exc:
                logger.warning(
                    "Attempt %d/%d: schema validation failed -- %s",
                    attempt, self._max_retries, exc.errors(),
                )
                if attempt == self._max_retries:
                    raise LLMInvalidResponseError(
                        f"Failed to generate a valid response after {self._max_retries} attempts."
                    ) from exc

                # Append the invalid response and correction instructions to the conversation
                messages.append({"role": "assistant", "content": raw_response})
                messages.append({
                    "role": "user",
                    "content": self._prompt_builder.build_correction_message(exc.json()),
                })

        # Unreachable, but satisfies static type checkers
        raise LLMInvalidResponseError("Analysis pipeline exited without a result.")
