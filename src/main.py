from contextlib import asynccontextmanager

import uvicorn
from fastapi import APIRouter, FastAPI

from controllers.api.health_controller import router as health_router
from core.consumer import BaseConsumer
from services.gemini_client import GeminiClientService
from services.qdrant_client import QdrantClientService
from utils.config import get_settings
from utils.logger import get_logger, setup_logging
from utils.server import get_version as get_server_version

# Load application settings and logging
setup_logging()
settings = get_settings()
application_version = get_server_version()
logger = get_logger(__name__)


# Application lifecycle manager
@asynccontextmanager
async def lifespan(_app: FastAPI):
    _app.state.amqp_url = settings.amqp_url
    _app.state.qdrant = QdrantClientService()
    _app.state.gemini = GeminiClientService()

    # Each consumer gets its own connection and channel (no shared conn on failure)
    consumers: list[BaseConsumer] = []

    logger.info("Starting %d consumer(s)", len(consumers))
    for consumer in consumers:
        consumer.start()
        logger.info("Started consumer: %s", consumer.__class__.__name__)

    yield

    logger.info("Stopping %d consumer(s)", len(consumers))
    for consumer in consumers:
        consumer.stop()
        logger.info("Stopped consumer: %s", consumer.__class__.__name__)


# Create FastAPI application
app = FastAPI(
    title=settings.server_name,
    version=application_version,
    lifespan=lifespan,
)

api_router = APIRouter(prefix="/api")
api_router.include_router(health_router)

app.include_router(api_router)


# Main function
def main() -> None:
    host = "0.0.0.0"
    port = settings.server_port
    logger.info(
        "Application %s listening on %s:%d",
        settings.server_name,
        host,
        port,
    )
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=settings.environment == "development",
    )


# Entry point
if __name__ == "__main__":
    main()
