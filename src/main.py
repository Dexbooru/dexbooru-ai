import asyncio
from contextlib import asynccontextmanager

import uvicorn
from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from consumers import CONSUMER_CLASSES
from controllers.api.health_controller import router as health_router
from controllers.api.similarity_posts_images_config_controller import router as similarity_post_images_config_router
from controllers.api.similarity_posts_images_controller import router as similarity_posts_images_router
from controllers.api.tag_rating_controller import router as tag_rating_router
from core.consumer import BaseConsumer
from core.database import Database
from core.llm import build_openai_compatible_model
from core.redis_client import close as close_redis
from core.redis_client import create_redis
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

    # Engines and pools connect lazily, so startup does not depend on Postgres or Redis being reachable
    database = Database(settings)
    _app.state.database = database
    logger.info(
        "Configured database pool pool_size=%d max_overflow=%d",
        settings.database_pool_size,
        settings.database_max_overflow,
    )
    redis = create_redis(settings)
    _app.state.redis = redis
    logger.info("Configured redis pool max_connections=%d", settings.redis_max_connections)
    _app.state.chat_model = build_openai_compatible_model(
        base_url=settings.chat_model_base_url,
        api_key=settings.chat_model_api_key,
        model_name=settings.chat_model_name,
        timeout=settings.chat_model_timeout_seconds,
    )
    logger.info("Configured chat model %s", settings.chat_model_name)

    consumers: list[BaseConsumer] = []
    try:
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
        consumers = [
            consumer_cls(
                amqp_url=settings.amqp_url,
                exchange_name=settings.primary_exchange_name,
                event_loop=loop,
                qdrant=_app.state.qdrant,
                gemini=_app.state.gemini,
            )
            for consumer_cls in CONSUMER_CLASSES
        ]

        logger.info("Starting %d consumer(s)", len(consumers))
        for consumer in consumers:
            consumer.start()
            logger.info("Started consumer: %s", consumer.__class__.__name__)

        yield
    finally:
        try:
            logger.info("Stopping %d consumer(s)", len(consumers))
            for consumer in consumers:
                consumer.stop()
                logger.info("Stopped consumer: %s", consumer.__class__.__name__)
        finally:
            try:
                await database.dispose()
                logger.info("Disposed database engine")
            finally:
                await close_redis(redis)
                logger.info("Closed redis client")


# Create FastAPI application
app = FastAPI(
    title=settings.server_name,
    version=application_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=not any(o == "*" for o in settings.cors_origins),
    allow_methods=["*"],
    allow_headers=["*"],
)

api_router = APIRouter(prefix="/api")
api_router.include_router(health_router)
api_router.include_router(similarity_post_images_config_router)
api_router.include_router(similarity_posts_images_router)
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
