"""
InputParser -- Stage 1: input normalisation and keyword extraction.

Separated from ContextRetriever (Stage 2) to honour the Single Responsibility
Principle: parsing is one concern, retrieval is another.

The interface is intentionally minimal so it can be replaced with a more
sophisticated NLP tokeniser (spaCy, NLTK) without touching the retriever
or the pipeline orchestrator.
"""
import logging
from typing import Any

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


class InputParser:
    """
    Responsible for Stage 1 of the analysis pipeline:
      parse_input -- normalise text and extract keywords.

    Minimum token length is set to 2 to preserve short-but-critical technical
    terms such as 'auth', 'smtp', 'cpu', 'api', '5xx', 'jwt'.
    """

    _MIN_KEYWORD_LENGTH = 2

    def parse_input(self, text: str) -> dict[str, Any]:
        """
        Normalise text and extract keywords.

        Returns a dict with ``raw_text`` and ``keywords`` for use by
        ContextRetriever.retrieve().
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
