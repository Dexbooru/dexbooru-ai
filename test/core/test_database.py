"""Unit tests for core.database.Database."""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.database import Database
from utils.config import get_settings


def _make_database() -> tuple[Database, MagicMock, MagicMock, MagicMock]:
    engine = MagicMock()
    engine.dispose = AsyncMock()
    session = MagicMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session_factory = MagicMock()
    session_factory.return_value.__aenter__ = AsyncMock(return_value=session)
    session_factory.return_value.__aexit__ = AsyncMock(return_value=False)
    with (
        patch("core.database.create_async_engine", return_value=engine) as create_engine,
        patch("core.database.async_sessionmaker", return_value=session_factory) as sessionmaker,
    ):
        database = Database(get_settings())
    create_engine.assert_called_once()
    sessionmaker.assert_called_once_with(engine, expire_on_commit=False)
    return database, engine, session_factory, session


class TestDatabaseInit:
    def test_engine_built_from_settings(self) -> None:
        settings = get_settings().model_copy(
            update={
                "database_pool_size": 7,
                "database_max_overflow": 3,
                "database_pool_timeout_seconds": 11,
                "database_pool_recycle_seconds": 900,
            }
        )
        with (
            patch("core.database.create_async_engine") as create_engine,
            patch("core.database.async_sessionmaker"),
        ):
            database = Database(settings)

        create_engine.assert_called_once_with(
            settings.database_url,
            pool_size=7,
            max_overflow=3,
            pool_timeout=11,
            pool_recycle=900,
            pool_pre_ping=True,
        )
        assert database.engine is create_engine.return_value

    def test_real_engine_uses_asyncpg_without_connecting(self) -> None:
        database = Database(get_settings())
        try:
            assert database.engine.dialect.driver == "asyncpg"
            assert database.engine.pool.size() == get_settings().database_pool_size
        finally:
            asyncio.run(database.dispose())


class TestDatabaseSession:
    def test_commits_on_success(self) -> None:
        database, _, _, session = _make_database()

        async def run() -> None:
            async with database.session() as yielded:
                assert yielded is session

        asyncio.run(run())

        session.commit.assert_awaited_once()
        session.rollback.assert_not_awaited()

    def test_rolls_back_and_reraises_on_error(self) -> None:
        database, _, _, session = _make_database()

        async def run() -> None:
            async with database.session():
                raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            asyncio.run(run())

        session.rollback.assert_awaited_once()
        session.commit.assert_not_awaited()

    def test_rolls_back_when_commit_fails(self) -> None:
        database, _, _, session = _make_database()
        session.commit.side_effect = RuntimeError("commit failed")

        async def run() -> None:
            async with database.session():
                pass

        with pytest.raises(RuntimeError, match="commit failed"):
            asyncio.run(run())

        session.rollback.assert_awaited_once()

    def test_rolls_back_on_cancellation(self) -> None:
        database, _, _, session = _make_database()

        async def run() -> None:
            async with database.session():
                raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(run())

        session.rollback.assert_awaited_once()


class TestDatabaseHealth:
    def test_is_healthy_true_when_select_succeeds(self) -> None:
        database, engine, _, _ = _make_database()
        connection = MagicMock()
        connection.execute = AsyncMock()
        engine.connect.return_value.__aenter__ = AsyncMock(return_value=connection)
        engine.connect.return_value.__aexit__ = AsyncMock(return_value=False)

        assert asyncio.run(database.is_healthy()) is True
        executed = connection.execute.await_args.args[0]
        assert str(executed) == "SELECT 1"

    def test_is_healthy_false_and_logs_without_url(self, caplog: pytest.LogCaptureFixture) -> None:
        database, engine, _, _ = _make_database()
        engine.connect.return_value.__aenter__ = AsyncMock(side_effect=OSError(f"cannot reach {get_settings().database_url}"))
        engine.connect.return_value.__aexit__ = AsyncMock(return_value=False)

        with caplog.at_level(logging.WARNING, logger="app.core.database"):
            assert asyncio.run(database.is_healthy()) is False

        assert "OSError" in caplog.text
        assert get_settings().database_url not in caplog.text
        assert "test:test" not in caplog.text

    def test_dispose_disposes_engine(self) -> None:
        database, engine, _, _ = _make_database()
        asyncio.run(database.dispose())
        engine.dispose.assert_awaited_once()
