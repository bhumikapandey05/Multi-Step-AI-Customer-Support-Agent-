"""Application settings loaded from environment variables (prefix ``SUPPORT_AGENT_``)."""

from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url

DEV_SECRET_KEY = "dev-only-insecure-secret-key-change-me-please"
MIN_SECRET_LENGTH = 32
# Secrets that are published in this repository (the built-in default and the old
# .env.example placeholder). Anyone could forge tokens with them, so production refuses them.
PUBLIC_SECRETS = frozenset({DEV_SECRET_KEY, "change-me-to-a-long-random-string-of-48-chars"})
_PLACEHOLDER_MARKERS = ("change-me", "changeme", "placeholder", "your-secret", "example")
# A free-tier Gemini Flash model. Override with GEMINI_MODEL (e.g. gemini-3.5-flash-lite
# for higher free-tier quotas).
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SUPPORT_AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    environment: Literal["development", "test", "production"] = "development"
    database_url: str = "sqlite:///data/crm.db"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["text", "json"] = "text"
    auth_secret_key: SecretStr = SecretStr(DEV_SECRET_KEY)
    session_ttl_minutes: int = Field(default=30, ge=1, le=24 * 60)
    pin_hash_iterations: int = Field(default=600_000, ge=1_000)

    # --- LLM (Phase 2). The key is read from GEMINI_API_KEY, the conventional name.
    gemini_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("GEMINI_API_KEY", "SUPPORT_AGENT_GEMINI_API_KEY"),
    )
    gemini_model: str = Field(
        default=DEFAULT_GEMINI_MODEL,
        validation_alias=AliasChoices("GEMINI_MODEL", "SUPPORT_AGENT_GEMINI_MODEL"),
    )
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    llm_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    llm_max_retries: int = Field(default=3, ge=0, le=10)

    # --- Agent loop guardrails
    agent_max_llm_rounds: int = Field(default=6, ge=1, le=20)
    agent_max_tool_calls_per_turn: int = Field(default=8, ge=1, le=50)
    agent_approval_ttl_minutes: int = Field(default=10, ge=1, le=120)
    agent_require_ticket_approval: bool = True

    # --- Phase 3: LangGraph workflows
    # LangGraph checkpoints (conversations, pending approvals) live in their own SQLite file.
    # Default: ``checkpoints.db`` next to the CRM database file.
    checkpoint_db_path: str | None = None
    # Bounded retries for transient CRM/database failures inside workflow steps.
    workflow_max_retries: int = Field(default=3, ge=0, le=10)
    workflow_retry_base_delay_seconds: float = Field(default=0.2, ge=0, le=10)
    staff_session_ttl_minutes: int = Field(default=60, ge=1, le=12 * 60)

    @field_validator("auth_secret_key", mode="before")
    @classmethod
    def _blank_secret_uses_default(cls, value: Any) -> Any:
        # A blank SUPPORT_AGENT_AUTH_SECRET_KEY (as in .env.example) means "not set": the
        # development default is used locally, and production refuses to start.
        return DEV_SECRET_KEY if isinstance(value, str) and not value.strip() else value

    @field_validator("gemini_api_key", mode="before")
    @classmethod
    def _blank_key_is_missing(cls, value: Any) -> Any:
        return None if isinstance(value, str) and not value.strip() else value

    @model_validator(mode="after")
    def _validate_secret(self) -> "Settings":
        secret = self.auth_secret_key.get_secret_value()
        if len(secret) < MIN_SECRET_LENGTH:
            raise ValueError(
                f"SUPPORT_AGENT_AUTH_SECRET_KEY must be at least {MIN_SECRET_LENGTH} characters"
            )
        if self.environment == "production" and self.uses_public_secret:
            raise ValueError(
                "A private SUPPORT_AGENT_AUTH_SECRET_KEY is required in production; the "
                "development default and placeholder values are public. Generate one with: "
                'python -c "import secrets; print(secrets.token_urlsafe(48))"'
            )
        return self

    @property
    def checkpoint_path(self) -> Path:
        if self.checkpoint_db_path:
            return Path(self.checkpoint_db_path)
        url = make_url(self.database_url)
        if url.get_backend_name() == "sqlite" and url.database not in (None, "", ":memory:"):
            return Path(url.database).with_name("checkpoints.db")
        return Path("data/checkpoints.db")

    @property
    def uses_public_secret(self) -> bool:
        """True for the built-in key or any placeholder-looking value: anyone could forge tokens."""
        secret = self.auth_secret_key.get_secret_value()
        return secret in PUBLIC_SECRETS or any(m in secret.lower() for m in _PLACEHOLDER_MARKERS)

    @property
    def uses_dev_secret(self) -> bool:
        return self.auth_secret_key.get_secret_value() == DEV_SECRET_KEY
