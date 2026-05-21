"""
FastAPI application entry point.

Key architectural decisions:
  - lifespan context manager instead of module-level global code:
    resource initialisation and cleanup happen explicitly at startup/shutdown.
  - Settings are read from the centralised Settings object once at startup.
  - IncidentAnalyzer is created inside lifespan and stored in app.state,
    making the dependency explicit and allowing overrides in tests.
  - Typed exceptions (LLMUnavailableError / LLMInvalidResponseError) are
    translated into precise HTTP status codes (503 / 422).
  - Short-text validation raises HTTPException(400) explicitly in the route
    rather than via Pydantic field_validator -- this preserves the correct
    HTTP 400 (Bad Request) semantic, distinct from 422 (Unprocessable Entity).
"""
import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator, List

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.settings import Settings
from app.agent.analyzer import IncidentAnalyzer
from app.agent.llm_client import OpenAILLMClient
from app.agent.retriever import ContextRetriever
from app.agent.prompt_builder import PromptBuilder
from app.agent.exceptions import LLMUnavailableError, LLMInvalidResponseError

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
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
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
    )
    app.state.analyzer = IncidentAnalyzer(
        llm_client=llm_client,
        retriever=ContextRetriever(),
        prompt_builder=PromptBuilder(),
        max_retries=settings.max_retries,
    )
    logger.info("IncidentAnalyzer initialised (model: %s).", settings.llm_model_name)

    yield  # application handles requests

    logger.info("Server shutting down.")


# ---------------------------------------------------------------------------
# Application and routes
# ---------------------------------------------------------------------------

app = FastAPI(title="AI Incident Triage API", lifespan=lifespan)


def get_analyzer(request: Request) -> IncidentAnalyzer:
    """Dependency injection: returns the analyzer from application state."""
    return request.app.state.analyzer


class AnalyzeRequest(BaseModel):
    """Input contract for the analysis endpoint."""

    incident_text: str = Field(
        ...,
        description="Raw incident description or log excerpt to analyse.",
    )


@app.post("/api/v1/analyze")
async def analyze_incident(
    req: AnalyzeRequest,
    analyzer: IncidentAnalyzer = Depends(get_analyzer),
) -> dict:
    """
    Analyse an incident description and return a structured JSON response.

    HTTP status codes:
      400 -- text is too short to provide meaningful context;
      422 -- LLM failed to produce a valid response after all retries,
             or request body is missing a required field;
      503 -- LLM API is temporarily unavailable;
      500 -- unexpected internal error.
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

    try:
        result = await analyzer.analyze(text)
        logger.info("Incident analysed. Category: %s", result.get("category"))
        return result
    except LLMInvalidResponseError as exc:
        logger.error("LLM response validation error: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except LLMUnavailableError as exc:
        logger.error("LLM API unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error("Unexpected error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error.")


# Mount the static frontend last so it does not shadow API routes
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    _settings = Settings()
    uvicorn.run(app, host=_settings.host, port=_settings.port)
