"""Unit tests for core.repository.BaseRepository."""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

from core.repository import BaseRepository


class _ExampleRepository(BaseRepository):
    async def fetch(self) -> object:
        async with self._session() as session:
            return await session.scalar("SELECT 1")


class TestBaseRepository:
    def test_session_opened_per_method_call(self) -> None:
        events: list[str] = []
        session = MagicMock()

        async def scalar(_statement: str) -> int:
            events.append("query")
            return 1

        session.scalar = scalar

        @asynccontextmanager
        async def fake_session():  # noqa: ANN202
            events.append("open")
            yield session
            events.append("close")

        database = MagicMock()
        database.session.side_effect = fake_session
        repository = _ExampleRepository(database)

        assert asyncio.run(repository.fetch()) == 1
        assert asyncio.run(repository.fetch()) == 1

        assert database.session.call_count == 2
        assert events == ["open", "query", "close", "open", "query", "close"]
