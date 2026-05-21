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

# Generic English stop words that carry no incident-specific signal.
# Technical terms (auth, smtp, cpu, etc.) are intentionally excluded from this
# set so they are preserved as keywords even though they are short.
# Domain-level generic nouns (e.g. "provider", "service", "error", "errors")
# are also excluded from keyword matching because they match too many incidents
# and reduce retrieval precision.
_STOP_WORDS = frozenset({
    # Common English function words
    "the", "and", "for", "are", "but", "not", "you", "all", "can",
    "had", "her", "was", "one", "our", "out", "day", "get", "has",
    "him", "his", "how", "its", "may", "new", "now", "old", "see",
    "two", "way", "who", "boy", "did", "its", "let", "put", "say",
    "she", "too", "use", "via", "from", "that", "this", "with",
    "have", "will", "your", "they", "been", "were", "when", "what",
    "some", "than", "then", "also", "into", "more", "very", "show",
    "logs", "log",
    # Short function words not caught by min-length filter
    "is", "on", "to", "in", "at", "by", "as", "an", "or", "if",
    "so", "do", "up", "no", "go", "it", "be", "we", "me", "my",
    # Domain-level generic terms -- appear across many incidents and add noise
    "provider", "service", "error", "errors", "issue", "issues",
})


class ContextRetriever:
    """
    Responsible for the first two pipeline stages:
      1. parse_input  -- normalise and tokenise the raw incident text;
      2. retrieve     -- select relevant past incidents.
    """

    # Minimum token length for a keyword to be considered meaningful.
    # Set to 2 to preserve short-but-critical technical terms such as
    # 'auth', 'smtp', 'cpu', 'api', '5xx', 'jwt'.
    _MIN_KEYWORD_LENGTH = 2

    def __init__(self, incidents: Optional[List[str]] = None) -> None:
        # Injecting the incident list makes unit testing with a custom dataset easy
        self._incidents = incidents if incidents is not None else PAST_INCIDENTS_LIST

    def parse_input(self, text: str) -> Dict[str, Any]:
        """
        Stage 1: normalise text and extract keywords.

        Returns a dict with raw_text and keywords for use by retrieve().
        Tokens are lower-cased and filtered against a stop-word list.
        """
        cleaned = text.strip()
        tokens = cleaned.replace("-", " ").lower().split()
        keywords = [
            t for t in tokens
            if len(t) >= self._MIN_KEYWORD_LENGTH and t not in _STOP_WORDS
        ]
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
