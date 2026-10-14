"""Async SQLAlchemy engine and session factory for the local Postgres replica.

Connection pooling rules:

- ``uvicorn workers * (pool_size + max_overflow)``, plus the replication apply worker and the
  table-sync workers, must stay below the replica's ``max_connections``.
- Never keep a database session open across an ``await`` on the chat model or a response stream.
  A reply can stream for tens of seconds; holding a connection that long drains the pool after a
  handful of concurrent chats. Repositories enforce this by opening a session per method call.
- If PgBouncer is added later in transaction mode, set asyncpg's ``statement_cache_size=0``
  (``connect_args={"statement_cache_size": 0}``). The logical replication subscription must always
  connect to Postgres directly, never through PgBouncer.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from utils.config import ApplicationSettings
from utils.logger import get_logger

logger = get_logger(__name__)


class Database:
    def __init__(self, settings: ApplicationSettings) -> None:
        self._engine: AsyncEngine = create_async_engine(
            settings.database_url,
            pool_size=settings.database_pool_size,
            max_overflow=settings.database_max_overflow,
            pool_timeout=settings.database_pool_timeout_seconds,
            pool_recycle=settings.database_pool_recycle_seconds,
            pool_pre_ping=True,
        )
        self._session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(self._engine, expire_on_commit=False)

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Yield a session that commits on success and rolls back (then re-raises) on any error."""
        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except BaseException:
                await session.rollback()
                raise

    async def is_healthy(self) -> bool:
        try:
            async with self._engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
            return True
        except Exception as error:
            logger.warning("Database health check failed error=%s", type(error).__name__)
            return False

    async def dispose(self) -> None:
        await self._engine.dispose()
