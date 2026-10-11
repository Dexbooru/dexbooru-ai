"""Shared async Redis client.

Create one client per process with ``create_redis`` and reuse it; every command borrows a connection
from the shared pool and returns it immediately. Each uvicorn worker owns its own pool, so
``uvicorn workers * redis_max_connections`` must stay below the Redis server's ``maxclients``.
"""

from redis.asyncio import ConnectionPool, Redis

from utils.config import ApplicationSettings
from utils.logger import get_logger

logger = get_logger(__name__)


def create_redis(settings: ApplicationSettings) -> Redis:
    pool = ConnectionPool.from_url(
        settings.redis_url,
        max_connections=settings.redis_max_connections,
        decode_responses=True,
    )
    return Redis(connection_pool=pool)


async def is_healthy(redis: Redis) -> bool:
    try:
        return bool(await redis.ping())
    except Exception as error:
        logger.warning("Redis health check failed error=%s", type(error).__name__)
        return False


async def close(redis: Redis) -> None:
    try:
        await redis.aclose()
    finally:
        await redis.connection_pool.disconnect()
