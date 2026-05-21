"""
ContextRetriever -- relevant context retrieval stage (RAG placeholder).

The current implementation uses keyword overlap for simplicity.
The interface is intentionally isolated so that the keyword approach can be
replaced with vector search (pgvector / FAISS) in the future without touching
the rest of the pipeline.
"""
import logging
from typing import Dict, Any, List, Optional

from app.context import PAST_INCIDENTS_LIST

logger = logging.getLogger(__name__)


class ContextRetriever:
    """
    Responsible for the first two pipeline stages:
      1. parse_input  -- normalise and tokenise the raw incident text;
      2. retrieve     -- select relevant past incidents.
    """

    # Stop words excluded when building the keyword set
    _STOP_WORDS = frozenset({"which", "their", "these", "those", "about", "would", "could"})

    def __init__(self, incidents: Optional[List[str]] = None) -> None:
        # Injecting the incident list makes unit testing with a custom dataset easy
        self._incidents = incidents if incidents is not None else PAST_INCIDENTS_LIST

    def parse_input(self, text: str) -> Dict[str, Any]:
        """
        Stage 1: normalise text and extract keywords.

        Returns a dict with raw_text and keywords for use by retrieve().
        """
        cleaned = text.strip()
        tokens = cleaned.replace("-", " ").lower().split()
        keywords = [t for t in tokens if len(t) > 4 and t not in self._STOP_WORDS]
        logger.debug("Extracted %d keywords from input text.", len(keywords))
        return {"raw_text": cleaned, "keywords": keywords}

    def retrieve(self, parsed_data: Dict[str, Any]) -> str:
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
