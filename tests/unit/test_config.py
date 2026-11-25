import pytest
from pydantic import ValidationError

from support_agent.config import DEV_SECRET_KEY, Settings


def test_defaults_are_usable_in_development():
    settings = Settings(_env_file=None)
    assert settings.environment == "development"
    assert settings.uses_dev_secret


def test_production_refuses_dev_secret():
    with pytest.raises(ValidationError, match="production"):
        Settings(_env_file=None, environment="production", auth_secret_key=DEV_SECRET_KEY)


@pytest.mark.parametrize(
    "secret",
    [
        "",  # blank, as in .env.example
        "change-me-to-a-long-random-string-of-48-chars",  # the old published placeholder
        "my-placeholder-secret-value-that-is-long-enough",
        "CHANGE-ME-please-this-is-at-least-32-characters",
    ],
)
def test_production_refuses_public_or_placeholder_secrets(secret):
    with pytest.raises(ValidationError, match="private SUPPORT_AGENT_AUTH_SECRET_KEY"):
        Settings(_env_file=None, environment="production", auth_secret_key=secret)


def test_production_accepts_a_generated_secret():
    import secrets

    settings = Settings(
        _env_file=None, environment="production", auth_secret_key=secrets.token_urlsafe(48)
    )
    assert not settings.uses_dev_secret


def test_blank_secret_means_development_default(monkeypatch):
    monkeypatch.setenv("SUPPORT_AGENT_AUTH_SECRET_KEY", "")
    settings = Settings(_env_file=None)
    assert settings.uses_dev_secret and settings.uses_public_secret


def test_placeholder_secret_is_flagged_as_public_in_development():
    settings = Settings(
        _env_file=None, auth_secret_key="change-me-to-a-long-random-string-of-48-chars"
    )
    assert settings.uses_public_secret and not settings.uses_dev_secret


def test_short_secret_is_refused():
    with pytest.raises(ValidationError, match="at least"):
        Settings(_env_file=None, auth_secret_key="short")


def test_env_vars_are_read(monkeypatch):
    monkeypatch.setenv("SUPPORT_AGENT_DATABASE_URL", "sqlite:///elsewhere.db")
    monkeypatch.setenv("SUPPORT_AGENT_SESSION_TTL_MINUTES", "5")
    settings = Settings(_env_file=None)
    assert settings.database_url == "sqlite:///elsewhere.db"
    assert settings.session_ttl_minutes == 5


def test_gemini_settings_use_conventional_env_names(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-value")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    settings = Settings(_env_file=None)
    assert settings.gemini_api_key.get_secret_value() == "test-key-value"
    assert settings.gemini_model == "gemini-3.5-flash-lite"
    assert "test-key-value" not in repr(settings)


def test_blank_gemini_key_counts_as_missing(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "  ")
    assert Settings(_env_file=None).gemini_api_key is None
