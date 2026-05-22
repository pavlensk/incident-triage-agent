"""
Unit tests for InputParser -- Stage 1 of the analysis pipeline.

InputParser is tested in isolation here.  Functional coverage through
test_agent.py (parser + retriever together) is not a substitute for
component isolation: it cannot pinpoint which stage introduced a regression.

Test classes:
  TestParseInputOutputContract  -- return shape and raw_text preservation
  TestMinKeywordLength          -- _MIN_KEYWORD_LENGTH = 2 boundary behaviour
  TestStopWordFiltering         -- function words and domain-generic terms
  TestHyphenNormalisation       -- replace("-", " ") pre-processing
  TestEdgeCases                 -- empty string, whitespace-only, punctuation
"""
import pytest

from app.agent.input_parser import InputParser, _STOP_WORDS


@pytest.fixture
def parser() -> InputParser:
    return InputParser()


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

class TestParseInputOutputContract:
    """parse_input() must always return a dict with 'raw_text' and 'keywords'."""

    def test_returns_raw_text_key(self, parser):
        result = parser.parse_input("auth service returning 401 errors")
        assert "raw_text" in result

    def test_returns_keywords_key(self, parser):
        result = parser.parse_input("auth service returning 401 errors")
        assert "keywords" in result

    def test_keywords_is_a_list(self, parser):
        result = parser.parse_input("auth service returning 401 errors")
        assert isinstance(result["keywords"], list)

    def test_raw_text_is_stripped_input(self, parser):
        """Leading/trailing whitespace is stripped; interior whitespace is kept."""
        result = parser.parse_input("  auth errors on pods  ")
        assert result["raw_text"] == "auth errors on pods"

    def test_raw_text_is_not_lowercased(self, parser):
        """raw_text must preserve the original casing for display purposes."""
        text = "PayGate timeout at 14:00 UTC"
        result = parser.parse_input(text)
        assert result["raw_text"] == text

    def test_keywords_are_lowercased(self, parser):
        """Keywords are always lower-cased for case-insensitive matching."""
        result = parser.parse_input("PayGate PostgreSQL SMTP")
        assert all(k == k.lower() for k in result["keywords"])


# ---------------------------------------------------------------------------
# _MIN_KEYWORD_LENGTH = 2 boundary
# ---------------------------------------------------------------------------

class TestMinKeywordLength:
    """
    The minimum keyword length is 2 to preserve short but critical technical
    terms (auth, cpu, api, 5xx, jwt).  Single-character tokens are always
    discarded regardless of stop-word membership.
    """

    @pytest.mark.parametrize("term", ["auth", "cpu", "api", "jwt", "5xx", "db"])
    def test_two_char_and_longer_technical_terms_are_kept(self, parser, term):
        """Terms at or above _MIN_KEYWORD_LENGTH that are not stop words survive."""
        result = parser.parse_input(f"the {term} component failed")
        assert term in result["keywords"], (
            f"Technical term '{term}' (len={len(term)}) must not be filtered "
            f"by _MIN_KEYWORD_LENGTH={InputParser._MIN_KEYWORD_LENGTH}"
        )

    def test_single_char_tokens_are_discarded(self, parser):
        """Single-character tokens are below the minimum and must be dropped."""
        result = parser.parse_input("a b c x y z incident occurred")
        single_chars = [k for k in result["keywords"] if len(k) == 1]
        assert single_chars == [], (
            f"Single-character tokens leaked into keywords: {single_chars}"
        )

    def test_exactly_two_char_token_is_kept_when_not_a_stop_word(self, parser):
        """Length-2 tokens that are not stop words must pass the filter."""
        # '5x' is not a stop word and is exactly 2 chars
        result = parser.parse_input("5x increase in latency observed")
        assert "5x" in result["keywords"]


# ---------------------------------------------------------------------------
# Stop-word filtering
# ---------------------------------------------------------------------------

class TestStopWordFiltering:
    """
    Two categories of stop words are defined:
      1. Common English function words (the, and, for, …)
      2. Domain-generic nouns that match too many incidents (provider, service, …)

    The second category is the critical one: without it, an SMTP query would
    retrieve the PayGate incident because both mention "provider".
    """

    @pytest.mark.parametrize("word", [
        "the", "and", "for", "are", "with", "from", "that", "this",
        "have", "will", "been", "when", "what", "some",
    ])
    def test_common_function_words_are_excluded(self, parser, word):
        query = f"system has {word} issue in production environment"
        result = parser.parse_input(query)
        assert word not in result["keywords"], (
            f"Function word '{word}' leaked into keywords -- add it to _STOP_WORDS"
        )

    @pytest.mark.parametrize("word", ["provider", "service", "error", "errors",
                                       "issue", "issues"])
    def test_domain_generic_terms_are_excluded(self, parser, word):
        """
        Domain-generic nouns match too many incidents and reduce retrieval
        precision.  They must be in _STOP_WORDS and filtered out.
        """
        query = f"payment {word} causing failures across all regions"
        result = parser.parse_input(query)
        assert word not in result["keywords"], (
            f"Domain-generic term '{word}' leaked into keywords -- it matches "
            "too many incidents and degrades retrieval precision."
        )

    def test_stop_words_set_is_non_empty(self):
        """Sanity check: _STOP_WORDS must be populated and exported."""
        assert len(_STOP_WORDS) > 0

    def test_technical_terms_are_not_stop_words(self, parser):
        """
        Short technical terms must NOT be in _STOP_WORDS, even though they are
        short.  Filtering them would break retrieval for the most common incident
        patterns (auth failures, smtp issues, cpu saturation).
        """
        for term in ("auth", "smtp", "cpu", "api", "jwt", "401", "504"):
            assert term not in _STOP_WORDS, (
                f"Technical term '{term}' is in _STOP_WORDS -- this breaks "
                "keyword extraction for common incident patterns."
            )


# ---------------------------------------------------------------------------
# Hyphen normalisation
# ---------------------------------------------------------------------------

class TestHyphenNormalisation:
    """
    Hyphenated service names (auth-service, payment-service) are split on '-'
    before tokenisation.  This ensures 'auth' is extracted as a keyword even
    when the input uses the hyphenated form.
    """

    def test_hyphenated_name_yields_both_parts(self, parser):
        """'auth-service' -> 'auth' (keyword) + 'service' (stop word, filtered)."""
        result = parser.parse_input("auth-service is returning 401 errors")
        assert "auth" in result["keywords"]

    def test_hyphenated_part_that_is_stop_word_is_filtered(self, parser):
        """The 'service' half of 'auth-service' is a stop word and must be dropped."""
        result = parser.parse_input("auth-service is returning 401 errors")
        assert "service" not in result["keywords"]

    def test_multi_hyphen_term_is_split_correctly(self, parser):
        """'payment-service-db' splits into three tokens; each filtered independently."""
        result = parser.parse_input("payment-service-db saturation observed")
        # 'payment' and 'db' survive; 'service' is a stop word
        assert "payment" in result["keywords"]
        assert "db" in result["keywords"]
        assert "service" not in result["keywords"]

    def test_hyphen_in_raw_text_is_preserved(self, parser):
        """Hyphen splitting affects keywords only -- raw_text keeps the original form."""
        text = "auth-service returning 401"
        result = parser.parse_input(text)
        assert "auth-service" in result["raw_text"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Boundary inputs that the pipeline might realistically encounter."""

    def test_empty_string_returns_empty_keywords(self, parser):
        """An empty incident text produces an empty keyword list, not an error."""
        result = parser.parse_input("")
        assert result["raw_text"] == ""
        assert result["keywords"] == []

    def test_whitespace_only_returns_empty_keywords(self, parser):
        """Whitespace-only input strips to empty and yields no keywords."""
        result = parser.parse_input("   \t\n  ")
        assert result["raw_text"] == ""
        assert result["keywords"] == []

    def test_all_stop_words_returns_empty_keywords(self, parser):
        """If every token is a stop word, keywords must be empty (not crash)."""
        result = parser.parse_input("the and for are with from that this")
        assert result["keywords"] == []

    def test_duplicate_tokens_are_preserved(self, parser):
        """
        parse_input does not deduplicate -- that is intentional.
        Deduplication would lose frequency signal that a future vector-based
        retriever might use.
        """
        result = parser.parse_input("auth auth auth service failures")
        assert result["keywords"].count("auth") == 3

    def test_numeric_tokens_are_kept(self, parser):
        """HTTP status codes and port numbers are valid keywords."""
        result = parser.parse_input("receiving 401 and 504 status codes")
        assert "401" in result["keywords"]
        assert "504" in result["keywords"]
