"""Unit tests for core.redis_client."""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import fakeredis
import pytest
from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError

from core import redis_client
from utils.config import get_settings


class TestCreateRedis:
    def test_builds_client_on_shared_pool_from_settings(self) -> None:
        settings = get_settings().model_copy(update={"redis_max_connections": 13})
        client = redis_client.create_redis(settings)
        try:
            assert isinstance(client, Redis)
            pool = client.connection_pool
            assert pool.max_connections == 13
            assert pool.connection_kwargs["decode_responses"] is True
            assert pool.connection_kwargs["host"] == "localhost"
            assert pool.connection_kwargs["port"] == 6380
            assert pool.connection_kwargs["db"] == 0
            assert pool.connection_kwargs["password"] == "test"
        finally:
            asyncio.run(redis_client.close(client))


class TestIsHealthy:
    def test_true_when_ping_succeeds(self) -> None:
        client = fakeredis.FakeAsyncRedis(decode_responses=True)
        assert asyncio.run(redis_client.is_healthy(client)) is True

    def test_false_when_ping_fails_and_logs_without_url(self, caplog: pytest.LogCaptureFixture) -> None:
        client = MagicMock()
        client.ping = AsyncMock(side_effect=RedisConnectionError(f"cannot reach {get_settings().redis_url}"))

        with caplog.at_level(logging.WARNING, logger="app.core.redis_client"):
            assert asyncio.run(redis_client.is_healthy(client)) is False

        assert "ConnectionError" in caplog.text
        assert get_settings().redis_url not in caplog.text


class TestClose:
    def test_closes_client_and_disconnects_pool(self) -> None:
        client = MagicMock()
        client.aclose = AsyncMock()
        client.connection_pool.disconnect = AsyncMock()

        asyncio.run(redis_client.close(client))

        client.aclose.assert_awaited_once()
        client.connection_pool.disconnect.assert_awaited_once()

    def test_disconnects_pool_even_if_client_close_fails(self) -> None:
        client = MagicMock()
        client.aclose = AsyncMock(side_effect=RuntimeError("boom"))
        client.connection_pool.disconnect = AsyncMock()

        with pytest.raises(RuntimeError):
            asyncio.run(redis_client.close(client))

        client.connection_pool.disconnect.assert_awaited_once()

    def test_close_fakeredis_client(self) -> None:
        client = fakeredis.FakeAsyncRedis(decode_responses=True)

        async def run() -> None:
            await client.set("key", "value")
            assert await client.get("key") == "value"
            await redis_client.close(client)

        asyncio.run(run())
