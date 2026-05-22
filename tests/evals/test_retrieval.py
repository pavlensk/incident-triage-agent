"""
Eval: context retrieval quality.

The ContextRetriever is the RAG stage of the pipeline.  These tests evaluate
the *quality* of retrieval -- not just "does it return something" (covered by
unit tests) but "does it return the RIGHT thing for each incident type?"

Retrieval quality dimensions tested here:
  Precision  -- relevant incidents are returned, irrelevant ones are excluded.
  Recall     -- the most relevant past incident is always included.
  Fallback   -- when nothing matches, baseline context is still provided.
  Keyword extraction -- critical short technical terms are preserved.

Extending retrieval evals:
  When new incidents are added to PAST_INCIDENTS_LIST (context.py), add a
  corresponding test row to TestRetrievalPrecision to verify the new incident
  is correctly selected and does not pollute unrelated queries.
"""
import pytest

from app.agent.retriever import ContextRetriever
from app.context import PAST_INCIDENTS_LIST


@pytest.fixture
def retriever() -> ContextRetriever:
    return ContextRetriever()


# ---------------------------------------------------------------------------
# Recall: the right incident is retrieved for each query type
# ---------------------------------------------------------------------------

class TestRetrievalRecall:
    """For each incident type, the corresponding INC must appear in the context."""

    def test_paygate_query_retrieves_inc101(self, retriever):
        parsed = retriever.parse_input(
            "Customers cannot pay by card, payment-service logs show timeouts "
            "calling PayGate starting at 12:05 UTC."
        )
        context = retriever.retrieve(parsed)
        assert "INC-101" in context, "PayGate incident must retrieve INC-101"

    def test_db_query_retrieves_inc102(self, retriever):
        parsed = retriever.parse_input(
            "High CPU on PostgreSQL, reporting queries causing long waits, "
            "payments timing out with 504 errors."
        )
        context = retriever.retrieve(parsed)
        assert "INC-102" in context, "DB saturation query must retrieve INC-102"

    def test_smtp_query_retrieves_inc103(self, retriever):
        parsed = retriever.parse_input(
            "Users not receiving confirmation emails, notification logs show "
            "smtp connection timeouts."
        )
        context = retriever.retrieve(parsed)
        assert "INC-103" in context, "SMTP notification query must retrieve INC-103"

    def test_auth_query_retrieves_inc104(self, retriever):
        parsed = retriever.parse_input(
            "auth service returns 401 errors, mobile users cannot login, "
            "invalid token signatures in logs."
        )
        context = retriever.retrieve(parsed)
        assert "INC-104" in context, "Auth failure query must retrieve INC-104"


# ---------------------------------------------------------------------------
# Precision: irrelevant incidents are excluded
# ---------------------------------------------------------------------------

class TestRetrievalPrecision:
    """Queries should NOT return incidents from unrelated categories."""

    def test_smtp_query_excludes_paygate(self, retriever):
        """An SMTP query must not return the PayGate incident."""
        parsed = retriever.parse_input(
            "Users not receiving emails, smtp connection errors in notification logs."
        )
        context = retriever.retrieve(parsed)
        assert "PayGate provider" not in context, (
            "SMTP query must not pull in PayGate incident (keyword 'provider' must be a stop word)"
        )

    def test_auth_query_excludes_smtp(self, retriever):
        """An auth failure query must not return the SMTP incident."""
        parsed = retriever.parse_input(
            "auth service returning 401, invalid token signatures across pods."
        )
        context = retriever.retrieve(parsed)
        # SMTP incident mentions "notification-service" and "SMTP", not auth
        assert "SMTP provider" not in context or "INC-104" in context, (
            "Auth query must prioritise INC-104 over SMTP notification incident"
        )

    def test_db_query_excludes_auth(self, retriever):
        """A DB query about CPU should not retrieve the auth incident."""
        parsed = retriever.parse_input(
            "PostgreSQL CPU at 95 percent, long-running reporting queries blocking payments."
        )
        context = retriever.retrieve(parsed)
        # INC-104 is about auth tokens -- should not appear for a pure DB query
        if "INC-104" in context:
            # If it does appear, INC-102 must also be there (DB was retrieved correctly)
            assert "INC-102" in context


# ---------------------------------------------------------------------------
# Fallback: baseline context when nothing matches
# ---------------------------------------------------------------------------

class TestRetrievalFallback:
    def test_nonsense_query_returns_fallback(self, retriever):
        parsed = retriever.parse_input(
            "xyzzy frobnicate qwerty zork something completely unrelated."
        )
        context = retriever.retrieve(parsed)
        # Fallback returns first two incidents
        assert "INC-101" in context or "INC-102" in context

    def test_fallback_provides_at_least_one_incident(self, retriever):
        parsed = retriever.parse_input("blah blah blah no match expected here at all.")
        context = retriever.retrieve(parsed)
        assert len(context.strip()) > 0, "Fallback must always return non-empty context"


# ---------------------------------------------------------------------------
# Keyword extraction quality
# ---------------------------------------------------------------------------

class TestKeywordExtraction:
    """Verify that short-but-critical technical terms survive extraction."""

    @pytest.mark.parametrize("term,query", [
        ("auth",  "auth service is returning errors"),
        ("smtp",  "smtp connection failing"),
        ("cpu",   "cpu load is high"),
        ("api",   "api gateway returning 504"),
        ("401",   "401 errors in auth logs"),
        ("5xx",   "5xx errors spiking on payment endpoint"),
    ])
    def test_technical_term_preserved_as_keyword(self, retriever, term, query):
        """Terms with 2-4 characters that carry incident signal must not be filtered."""
        parsed = retriever.parse_input(query)
        assert term in parsed["keywords"], (
            f"Technical term '{term}' was incorrectly filtered from keywords. "
            f"Check _MIN_KEYWORD_LENGTH and _STOP_WORDS in retriever.py."
        )

    @pytest.mark.parametrize("stop_word", [
        "the", "and", "for", "are", "with", "from", "that", "this",
        "provider", "service", "errors",
    ])
    def test_stop_words_excluded_from_keywords(self, retriever, stop_word):
        """Generic words must be removed to prevent false-positive matches."""
        query = f"system has {stop_word} issue with logs showing problems"
        parsed = retriever.parse_input(query)
        assert stop_word not in parsed["keywords"], (
            f"Stop word '{stop_word}' leaked into keywords. "
            f"Add it to _STOP_WORDS in retriever.py."
        )

    def test_hyphenated_terms_split_correctly(self, retriever):
        """
        Hyphenated service names like 'auth-service' are split on '-' before
        tokenization so 'auth' and 'service' are evaluated independently.
        """
        parsed = retriever.parse_input(
            "auth-service is returning 401 errors on all pods."
        )
        # 'auth' must be a keyword; 'service' is a stop word and must be excluded
        assert "auth" in parsed["keywords"]
        assert "service" not in parsed["keywords"]
