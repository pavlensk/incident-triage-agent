"""
Centralised application configuration.

All parameters are read from environment variables (or a .env file) and
validated at startup -- no os.getenv() calls in business logic.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Single source of truth for application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # silently ignore unknown environment variables
    )

    # --- LLM: schema-validation retry loop ---
    openai_api_key: str = "dummy_key"
    llm_model_name: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    max_retries: int = 3          # max attempts to get a schema-valid response

    # --- LLM: transient-error retry policy ---
    llm_timeout_seconds: float = 30.0       # per-request HTTP timeout
    llm_retry_attempts: int = 2             # retries on rate-limit / connection errors
    llm_retry_delay_seconds: float = 1.0    # base delay; doubles on each attempt

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000

    # --- Logging ---
    log_level: str = "INFO"
    log_file: str = ""  # empty string means console-only logging
