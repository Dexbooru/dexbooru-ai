import asyncio

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from core.consumer import is_amqp_healthy
from core.dependencies import get_amqp_url, get_gemini, get_qdrant
from models.api_responses.health import HealthResponse
from services.gemini_client import GeminiClientService
from services.qdrant_client import QdrantClientService
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/", response_model=None)
async def get_health_check(
    qdrant: QdrantClientService = Depends(get_qdrant),
    gemini: GeminiClientService = Depends(get_gemini),
    amqp_url: str = Depends(get_amqp_url),
) -> HealthResponse | JSONResponse:
    qdrant_ok = False
    try:
        qdrant_ok = await qdrant.is_healthy()
        logger.info("Health check qdrant: %s", qdrant_ok)
    except Exception:
        logger.exception("Health check qdrant failed")

    gemini_ok = False
    try:
        gemini_ok = await asyncio.to_thread(gemini.is_healthy)
        logger.info("Health check gemini: %s", gemini_ok)
    except Exception:
        logger.exception("Health check gemini failed")

    amqp_ok = False
    try:
        amqp_ok = await asyncio.to_thread(is_amqp_healthy, amqp_url)
        logger.info("Health check amqp: %s", amqp_ok)
    except Exception:
        logger.exception("Health check amqp failed")

    response = HealthResponse(qdrant=qdrant_ok, gemini=gemini_ok, amqp=amqp_ok)
    if not (qdrant_ok and gemini_ok and amqp_ok):
        return JSONResponse(content=response.model_dump(), status_code=503)
    return response
