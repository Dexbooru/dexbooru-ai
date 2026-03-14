"""App-wide dependency injection. Services are created in lifespan and reused."""

from fastapi import Request

from services.gemini_client import GeminiClientService
from services.qdrant_client import QdrantClientService


def get_qdrant(request: Request) -> QdrantClientService:
    return request.app.state.qdrant


def get_gemini(request: Request) -> GeminiClientService:
    return request.app.state.gemini


def get_amqp_url(request: Request) -> str:
    return request.app.state.amqp_url
