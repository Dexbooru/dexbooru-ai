"""Pytest configuration. Ensures src is on the path for imports."""

import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
src = root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))


_TEST_REQUIRED_SETTINGS_ENV = {
    "SERVER_NAME": "dexbooru-ai-test",
    "AMQP_URL": "amqp://localhost/",
    "QDRANT_URL": "http://localhost:6333",
    "QDRANT_API_KEY": "test-qdrant-key",
    "GEMINI_API_KEY": "test-gemini-key",
    "CDN_BASE_URL": "https://cdn.example.com",
    "DATABASE_URL": "postgresql+asyncpg://test:test@localhost:5433/dexbooru_ai",
    "REDIS_URL": "redis://:test@localhost:6380/0",
    "JWT_SECRET": "test-jwt-secret",
    "APP_DOMAIN": "localhost",
    "CHAT_MODEL_BASE_URL": "http://localhost:11434/v1",
    "CHAT_MODEL_API_KEY": "test-chat-key",
    "CHAT_MODEL_NAME": "test-chat-model",
}


def pytest_configure(config) -> None:
    """Ensure required env vars exist before ApplicationSettings is first loaded."""
    skops = root / "models" / "danbooru_tag_rating_predictor.skops"
    os.environ.setdefault("DANBOORU_TAG_RATING_SKOPS_PATH", str(skops))
    for name, value in _TEST_REQUIRED_SETTINGS_ENV.items():
        os.environ.setdefault(name, value)
    import utils.config as config_module

    reset = getattr(config_module, "reset_settings_cache", None)
    if reset is not None:
        reset()
    else:
        setattr(config_module, "_instance", None)
