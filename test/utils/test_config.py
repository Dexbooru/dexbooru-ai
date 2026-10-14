"""Unit tests for utils.config ApplicationSettings."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from utils.config import ApplicationSettings, get_settings, reset_settings_cache

_REQUIRED_CHAT_SETTINGS_ENV = {
    "DATABASE_URL": "postgresql+asyncpg://user:password@localhost:5433/dexbooru_ai",
    "REDIS_URL": "redis://:password@localhost:6380/0",
    "JWT_SECRET": "test-jwt-secret",
    "APP_DOMAIN": "localhost",
    "CHAT_MODEL_BASE_URL": "http://localhost:11434/v1",
    "CHAT_MODEL_API_KEY": "test-chat-key",
    "CHAT_MODEL_NAME": "test-chat-model",
}


def _set_required_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SERVER_NAME", "dexbooru-ai-test")
    monkeypatch.setenv("AMQP_URL", "amqp://localhost/")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("CDN_BASE_URL", "https://cdn.example.com")
    monkeypatch.setenv(
        "DANBOORU_TAG_RATING_SKOPS_PATH",
        str(Path(__file__).resolve().parents[2] / "models" / "danbooru_tag_rating_predictor.skops"),
    )
    for name, value in _REQUIRED_CHAT_SETTINGS_ENV.items():
        monkeypatch.setenv(name, value)


class TestAllowedImageDomainDestinations:
    """Tests for allowed_image_domain_destinations settings properties."""

    def setup_method(self) -> None:
        reset_settings_cache()

    def teardown_method(self) -> None:
        reset_settings_cache()

    def test_defaults_to_allow_all_when_env_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_required_settings_env(monkeypatch)
        monkeypatch.delenv("ALLOWED_IMAGE_DOMAIN_DESTINATIONS", raising=False)

        settings = ApplicationSettings(_env_file=None)

        assert settings.allowed_image_domain_destinations == ["*"]
        assert settings.allow_all_image_domain_destinations is True

    def test_parses_comma_separated_allowed_domains(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_required_settings_env(monkeypatch)
        monkeypatch.setenv("ALLOWED_IMAGE_DOMAIN_DESTINATIONS", "cdn.example.com, images.example.org")

        settings = ApplicationSettings(_env_file=None)

        assert settings.allowed_image_domain_destinations == [
            "cdn.example.com",
            "images.example.org",
        ]
        assert settings.allow_all_image_domain_destinations is False

    def test_allow_all_is_false_when_star_mixed_with_domains(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_required_settings_env(monkeypatch)
        monkeypatch.setenv("ALLOWED_IMAGE_DOMAIN_DESTINATIONS", "*,cdn.example.com")

        settings = ApplicationSettings(_env_file=None)

        assert settings.allowed_image_domain_destinations == ["*", "cdn.example.com"]
        assert settings.allow_all_image_domain_destinations is False

    def test_get_settings_returns_cached_singleton(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_required_settings_env(monkeypatch)
        monkeypatch.setenv("ALLOWED_IMAGE_DOMAIN_DESTINATIONS", "cdn.example.com")

        first = get_settings()
        second = get_settings()

        assert first is second
        assert first.allowed_image_domain_destinations == ["cdn.example.com"]


_DEFAULTED_SETTINGS = {
    "DATABASE_POOL_SIZE": ("database_pool_size", 10),
    "DATABASE_MAX_OVERFLOW": ("database_max_overflow", 5),
    "DATABASE_POOL_TIMEOUT_SECONDS": ("database_pool_timeout_seconds", 10),
    "DATABASE_POOL_RECYCLE_SECONDS": ("database_pool_recycle_seconds", 1800),
    "REDIS_MAX_CONNECTIONS": ("redis_max_connections", 20),
    "UPSTREAM_DATABASE_URL": ("upstream_database_url", None),
    "REPLICATION_PUBLICATION": ("replication_publication", "dexbooru_ai_content"),
    "REPLICATION_SUBSCRIPTION": ("replication_subscription", "dexbooru_ai_content_sub"),
    "REPLICATION_SLOT": ("replication_slot", "dexbooru_ai_content_slot"),
    "JWT_ISSUER": ("jwt_issuer", None),
    "JWT_AUDIENCE": ("jwt_audience", None),
    "TRUSTED_PROXY_IPS": ("trusted_proxy_ips", []),
    "CHAT_MODEL_TIMEOUT_SECONDS": ("chat_model_timeout_seconds", 60),
    "CHAT_REQUEST_LIMIT": ("chat_request_limit", 8),
    "CHAT_HISTORY_MESSAGE_LIMIT": ("chat_history_message_limit", 20),
    "CHAT_MAX_MESSAGE_CHARS": ("chat_max_message_chars", 2000),
    "CHAT_USER_MESSAGES_PER_MINUTE": ("chat_user_messages_per_minute", 6),
    "CHAT_IP_MESSAGES_PER_MINUTE": ("chat_ip_messages_per_minute", 20),
    "CHAT_USER_READS_PER_MINUTE": ("chat_user_reads_per_minute", 60),
    "AUTH_IP_REQUESTS_PER_MINUTE": ("auth_ip_requests_per_minute", 10),
    "CHAT_DAILY_THREAD_LIMIT": ("chat_daily_thread_limit", 10),
    "CHAT_DAILY_MESSAGE_LIMIT": ("chat_daily_message_limit", 100),
    "CHAT_DAILY_TOKEN_LIMIT": ("chat_daily_token_limit", 200000),
    "CHAT_THREAD_MESSAGE_LIMIT": ("chat_thread_message_limit", 40),
}


class TestChatSettings:
    """Tests for database, Redis, auth and chat settings."""

    def test_defaults_when_env_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_required_settings_env(monkeypatch)
        for name in _DEFAULTED_SETTINGS:
            monkeypatch.delenv(name, raising=False)

        settings = ApplicationSettings(_env_file=None)

        for field_name, expected in _DEFAULTED_SETTINGS.values():
            assert getattr(settings, field_name) == expected, field_name

    def test_loads_required_settings_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_required_settings_env(monkeypatch)

        settings = ApplicationSettings(_env_file=None)

        assert settings.database_url == _REQUIRED_CHAT_SETTINGS_ENV["DATABASE_URL"]
        assert settings.redis_url == _REQUIRED_CHAT_SETTINGS_ENV["REDIS_URL"]
        assert settings.jwt_secret == _REQUIRED_CHAT_SETTINGS_ENV["JWT_SECRET"]
        assert settings.app_domain == _REQUIRED_CHAT_SETTINGS_ENV["APP_DOMAIN"]
        assert settings.chat_model_base_url == _REQUIRED_CHAT_SETTINGS_ENV["CHAT_MODEL_BASE_URL"]
        assert settings.chat_model_api_key == _REQUIRED_CHAT_SETTINGS_ENV["CHAT_MODEL_API_KEY"]
        assert settings.chat_model_name == _REQUIRED_CHAT_SETTINGS_ENV["CHAT_MODEL_NAME"]

    @pytest.mark.parametrize("missing_env_name", list(_REQUIRED_CHAT_SETTINGS_ENV))
    def test_missing_required_setting_raises(self, monkeypatch: pytest.MonkeyPatch, missing_env_name: str) -> None:
        _set_required_settings_env(monkeypatch)
        monkeypatch.delenv(missing_env_name)

        with pytest.raises(ValidationError, match=missing_env_name):
            ApplicationSettings(_env_file=None)

    @pytest.mark.parametrize("env_name", ["DATABASE_POOL_SIZE", "REDIS_MAX_CONNECTIONS", "CHAT_DAILY_TOKEN_LIMIT"])
    @pytest.mark.parametrize("value", ["0", "-1"])
    def test_non_positive_limits_raise(self, monkeypatch: pytest.MonkeyPatch, env_name: str, value: str) -> None:
        _set_required_settings_env(monkeypatch)
        monkeypatch.setenv(env_name, value)

        with pytest.raises(ValidationError, match=env_name):
            ApplicationSettings(_env_file=None)

    def test_database_max_overflow_allows_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_required_settings_env(monkeypatch)
        monkeypatch.setenv("DATABASE_MAX_OVERFLOW", "0")

        settings = ApplicationSettings(_env_file=None)

        assert settings.database_max_overflow == 0

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("", []),
            ("   ", []),
            ("10.0.0.1", ["10.0.0.1"]),
            ("10.0.0.1, 172.16.0.0/12 ,,192.168.1.1", ["10.0.0.1", "172.16.0.0/12", "192.168.1.1"]),
        ],
    )
    def test_parses_trusted_proxy_ips(self, monkeypatch: pytest.MonkeyPatch, raw: str, expected: list[str]) -> None:
        _set_required_settings_env(monkeypatch)
        monkeypatch.setenv("TRUSTED_PROXY_IPS", raw)

        settings = ApplicationSettings(_env_file=None)

        assert settings.trusted_proxy_ips == expected
