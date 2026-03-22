import asyncio
from contextlib import asynccontextmanager

import uvicorn
from fastapi import APIRouter, FastAPI

from consumers.new_post_consumer import NewPostConsumer
from controllers.api.health_controller import router as health_router
from controllers.api.tag_rating_controller import router as tag_rating_router
from core.consumer import BaseConsumer
from ml.dexbooru_tag_rating_predictor import DexbooruTagRatingPredictor
from services.gemini_client import GeminiClientService
from services.qdrant_client import QdrantClientService
from services.spacy_nlp import load_spacy_english
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

    logger.info("Loading spaCy model %s", settings.spacy_english_model)
    _app.state.nlp = load_spacy_english(settings)
    logger.info("Loading Danbooru tag-rating skops from %s", settings.danbooru_tag_rating_skops_path)
    _app.state.tag_rating_predictor = DexbooruTagRatingPredictor(
        nlp=_app.state.nlp,
        skops_path=settings.danbooru_tag_rating_skops_path,
    )

    _app.state.qdrant = QdrantClientService()
    _app.state.gemini = GeminiClientService()

    # Initialize base collections, if they don't exist already
    base_collections_created_successfully = await _app.state.qdrant.create_base_collections()
    if not base_collections_created_successfully:
        logger.error("Failed to create base collections")
        raise RuntimeError("Failed to create base collections")

    loop = asyncio.get_running_loop()
    consumers: list[BaseConsumer] = [
        NewPostConsumer(
            amqp_url=settings.amqp_url,
            exchange_name=settings.primary_exchange_name,
            event_loop=loop,
            qdrant=_app.state.qdrant,
            gemini=_app.state.gemini,
        ),
    ]

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
api_router.include_router(tag_rating_router)

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
