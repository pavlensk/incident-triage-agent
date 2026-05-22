"""
ContextRetriever -- Stage 2: relevant context retrieval (RAG placeholder).

The current implementation uses keyword overlap for simplicity.
The interface is intentionally isolated so that the keyword approach can be
replaced with vector search (pgvector / FAISS) in the future without touching
the rest of the pipeline.
"""
import logging
from typing import Any

from app.context import PAST_INCIDENTS_LIST

logger = logging.getLogger(__name__)


class ContextRetriever:
    """
    Responsible for Stage 2 of the analysis pipeline:
      retrieve -- select relevant past incidents by keyword overlap.
    """

    def __init__(self, incidents: list[str] | None = None) -> None:
        # Injecting the incident list makes unit testing with a custom dataset easy
        self._incidents = incidents if incidents is not None else PAST_INCIDENTS_LIST

    def retrieve(self, parsed_data: dict[str, Any]) -> str:
        """
        Stage 2: find past incidents matching the keyword set.

        Falls back to the first two incidents when nothing matches,
        so the LLM always receives some baseline context.
        """
        keywords = parsed_data.get("keywords", [])
        relevant = [
            inc for inc in self._incidents
            if any(kw in inc.lower() for kw in keywords)
        ]
        if not relevant:
            logger.debug("No relevant incidents found, using fallback.")
            relevant = self._incidents[:2]
        logger.debug("Selected %d relevant incident(s).", len(relevant))
        return "\n".join(relevant)
