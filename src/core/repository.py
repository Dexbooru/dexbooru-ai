from contextlib import AbstractAsyncContextManager

from sqlalchemy.ext.asyncio import AsyncSession

from core.database import Database


class BaseRepository:
    """Base for repositories. Open a session with ``_session()`` inside each method and never return or store it."""

    def __init__(self, database: Database) -> None:
        self._database = database

    def _session(self) -> AbstractAsyncContextManager[AsyncSession]:
        return self._database.session()
