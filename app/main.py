"""
FastAPI application entry point.

Key architectural decisions:
  - lifespan context manager: resources are initialised and released
    explicitly at startup/shutdown, not at module import time.
  - Settings object is the single source of configuration truth; no
    os.getenv() calls in business logic.
  - IncidentAnalyzer is stored in app.state and injected via get_analyzer()
    dependency -- easy to override in tests.
  - All domain exceptions are mapped to HTTP status codes by registered
    global exception handlers, so route handlers stay thin:

      LLMAuthenticationError   -> 500  (server misconfiguration)
      LLMUnavailableError      -> 503  (transient service error)
      LLMRateLimitError        -> 503  (inherits LLMUnavailableError)
      LLMInvalidResponseError  -> 422  (LLM could not produce valid output)

  - Error responses have a consistent JSON structure:
      {"code": "<machine_readable>", "message": "<human_readable>"}
    HTTPException (400 / 422 validation) keeps FastAPI's default format
      {"detail": "..."} for compatibility with standard tooling.
"""
import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.settings import Settings
from app.agent.analyzer import IncidentAnalyzer
from app.agent.input_parser import InputParser
from app.agent.llm_client import OpenAILLMClient
from app.agent.retriever import ContextRetriever
from app.agent.prompt_builder import PromptBuilder
from app.agent.exceptions import (
    LLMAuthenticationError,
    LLMInvalidResponseError,
    LLMUnavailableError,
)

# Module-level logger: created once, reused on every request.
logger = logging.getLogger(__name__)

# Minimum requirements for a meaningful incident description.
_MIN_WORDS = 5
_MIN_CHARS = 30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logging(settings: Settings) -> None:
    """Configure the root logger based on application settings."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if settings.log_file:
        handlers.append(logging.FileHandler(settings.log_file))
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Manage the FastAPI application lifecycle.

    Everything that requires initialisation (logger, LLM client, analyzer)
    is created here -- not at module import time.
    This eliminates import-time side effects and simplifies testing.
    """
    settings = Settings()
    _setup_logging(settings)

    if settings.openai_api_key == "dummy_key":
        logger.warning("OPENAI_API_KEY is not set -- LLM calls will fail.")

    llm_client = OpenAILLMClient(
        api_key=settings.openai_api_key,
        model_name=settings.llm_model_name,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout_seconds,
        base_url=settings.openai_base_url,
    )
    app.state.analyzer = IncidentAnalyzer(
        llm_client=llm_client,
        input_parser=InputParser(),
        retriever=ContextRetriever(),
        prompt_builder=PromptBuilder(),
        max_retries=settings.max_retries,
        llm_retry_attempts=settings.llm_retry_attempts,
        llm_retry_delay_seconds=settings.llm_retry_delay_seconds,
    )
    logger.info("IncidentAnalyzer initialised (model: %s).", settings.llm_model_name)

    yield  # application handles requests

    logger.info("Server shutting down.")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(title="AI Incident Triage API", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Global exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(LLMAuthenticationError)
async def llm_auth_error_handler(
    request: Request, exc: LLMAuthenticationError
) -> JSONResponse:
    """
    Authentication failures indicate a server-side configuration problem
    (wrong or revoked API key).  Return 500 with a safe, non-leaking message.
    """
    logger.critical(
        "LLM authentication failure -- check OPENAI_API_KEY. Detail: %s", exc
    )
    return JSONResponse(
        status_code=500,
        content={
            "code": "llm_configuration_error",
            "message": "The LLM service is misconfigured. Contact the administrator.",
        },
    )


@app.exception_handler(LLMUnavailableError)
async def llm_unavailable_handler(
    request: Request, exc: LLMUnavailableError
) -> JSONResponse:
    """
    Transient LLM errors (connection, timeout, rate limit after backoff).
    LLMRateLimitError inherits LLMUnavailableError so it is caught here too.
    """
    logger.error("LLM service unavailable: %s", exc)
    return JSONResponse(
        status_code=503,
        content={
            "code": "llm_unavailable",
            "message": str(exc),
        },
    )


@app.exception_handler(LLMInvalidResponseError)
async def llm_invalid_response_handler(
    request: Request, exc: LLMInvalidResponseError
) -> JSONResponse:
    """LLM exhausted all schema-validation retries without producing valid output."""
    logger.error("LLM invalid response after all retries: %s", exc)
    return JSONResponse(
        status_code=422,
        content={
            "code": "llm_invalid_response",
            "message": str(exc),
        },
    )


# ---------------------------------------------------------------------------
# Dependency and request model
# ---------------------------------------------------------------------------

def get_analyzer(request: Request) -> IncidentAnalyzer:
    """Dependency injection: returns the analyzer from application state."""
    return request.app.state.analyzer


class AnalyzeRequest(BaseModel):
    """Input contract for the analysis endpoint."""

    incident_text: str = Field(
        ...,
        description="Raw incident description or log excerpt to analyse.",
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    """Liveness probe -- returns 200 when the server is running."""
    return {"status": "ok"}


@app.post("/api/v1/analyze")
async def analyze_incident(
    req: AnalyzeRequest,
    analyzer: IncidentAnalyzer = Depends(get_analyzer),
) -> dict:
    """
    Analyse an incident description and return a structured JSON response.

    HTTP status codes:
      200 -- analysis succeeded.
      400 -- incident_text is too short to provide meaningful context.
      422 -- request body is missing a required field (FastAPI/Pydantic),
             or the LLM exhausted all retries without a valid response.
      500 -- LLM API key is invalid or revoked (server misconfiguration).
      503 -- LLM API is temporarily unavailable or rate-limited.
    """
    text = req.incident_text.strip()
    if len(text.split()) < _MIN_WORDS or len(text) < _MIN_CHARS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Incident text is too short. "
                f"Please provide at least {_MIN_WORDS} words and {_MIN_CHARS} characters."
            ),
        )

    # Domain exceptions (LLMAuthenticationError, LLMUnavailableError,
    # LLMInvalidResponseError) propagate to their registered global handlers.
    # Only truly unexpected errors are caught here.
    try:
        result = await analyzer.analyze(text)
    except (LLMAuthenticationError, LLMUnavailableError, LLMInvalidResponseError):
        raise  # handled by global exception handlers above
    except Exception as exc:
        logger.error("Unexpected error during analysis: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

    logger.info("Incident analysed. Category: %s", result.category)
    return result.model_dump()


# Mount the static frontend last so it does not shadow API routes
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    _settings = Settings()
    uvicorn.run(app, host=_settings.host, port=_settings.port)
 host=_settings.host, port=_settings.port)
