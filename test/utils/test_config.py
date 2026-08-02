"""Unit tests for utils.config ApplicationSettings image-domain helpers."""

from pathlib import Path

import pytest

from utils.config import ApplicationSettings, get_settings, reset_settings_cache


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
