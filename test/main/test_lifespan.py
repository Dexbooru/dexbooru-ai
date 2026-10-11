"""Unit tests for main.lifespan resource setup and teardown."""

import asyncio
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI

import main
from utils.config import get_settings


@dataclass
class _Patched:
    database_cls: MagicMock
    create_redis: MagicMock
    close_redis: AsyncMock
    build_model: MagicMock
    consumers: list[MagicMock]
    qdrant: MagicMock


@contextmanager
def _patched_lifespan(collections_ok: bool = True) -> Iterator[_Patched]:
    database_cls = MagicMock()
    database_cls.return_value.dispose = AsyncMock()
    qdrant = MagicMock()
    qdrant.create_base_collections = AsyncMock(return_value=collections_ok)
    consumer = MagicMock()
    with (
        patch.object(main, "Database", database_cls),
        patch.object(main, "create_redis") as create_redis,
        patch.object(main, "close_redis", new_callable=AsyncMock) as close_redis,
        patch.object(main, "build_openai_compatible_model") as build_model,
        patch.object(main, "load_spacy_english"),
        patch.object(main, "DexbooruTagRatingPredictor"),
        patch.object(main, "QdrantClientService", return_value=qdrant),
        patch.object(main, "GeminiClientService"),
        patch.object(main, "CONSUMER_CLASSES", (MagicMock(return_value=consumer),)),
    ):
        yield _Patched(database_cls, create_redis, close_redis, build_model, [consumer], qdrant)


class TestLifespan:
    def test_startup_creates_resources_once_and_shutdown_releases_them(self, caplog: pytest.LogCaptureFixture) -> None:
        settings = get_settings()
        app = FastAPI()

        async def run(patched: _Patched) -> None:
            async with main.lifespan(app):
                assert app.state.database is patched.database_cls.return_value
                assert app.state.redis is patched.create_redis.return_value
                assert app.state.chat_model is patched.build_model.return_value
                for consumer in patched.consumers:
                    consumer.start.assert_called_once()
                patched.database_cls.return_value.dispose.assert_not_awaited()
                patched.close_redis.assert_not_awaited()

        with _patched_lifespan() as patched, caplog.at_level(logging.INFO, logger="app"):
            asyncio.run(run(patched))

        patched.database_cls.assert_called_once_with(main.settings)
        patched.create_redis.assert_called_once_with(main.settings)
        patched.build_model.assert_called_once_with(
            base_url=settings.chat_model_base_url,
            api_key=settings.chat_model_api_key,
            model_name=settings.chat_model_name,
            timeout=settings.chat_model_timeout_seconds,
        )
        patched.database_cls.return_value.dispose.assert_awaited_once()
        patched.close_redis.assert_awaited_once_with(patched.create_redis.return_value)
        for consumer in patched.consumers:
            consumer.stop.assert_called_once()

        assert "Configured database pool" in caplog.text
        assert "Configured redis pool" in caplog.text
        for secret in (settings.database_url, settings.redis_url, settings.chat_model_api_key, "test:test", ":test@"):
            assert secret not in caplog.text

    def test_cleanup_runs_when_startup_fails(self) -> None:
        app = FastAPI()

        async def run() -> None:
            async with main.lifespan(app):
                pass

        with _patched_lifespan(collections_ok=False) as patched:
            with pytest.raises(RuntimeError, match="base collections"):
                asyncio.run(run())

        patched.database_cls.return_value.dispose.assert_awaited_once()
        patched.close_redis.assert_awaited_once()
        for consumer in patched.consumers:
            consumer.start.assert_not_called()

    def test_redis_closed_even_if_database_dispose_fails(self) -> None:
        app = FastAPI()

        async def run() -> None:
            async with main.lifespan(app):
                pass

        with _patched_lifespan() as patched:
            patched.database_cls.return_value.dispose.side_effect = RuntimeError("dispose failed")
            with pytest.raises(RuntimeError, match="dispose failed"):
                asyncio.run(run())

        patched.close_redis.assert_awaited_once()
