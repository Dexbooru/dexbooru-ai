"""App-wide dependency injection. Services are created in lifespan and reused."""

from fastapi import Request

from core.consumer import BaseConsumer
from services.gemini_client import GeminiClientService
from services.qdrant_client import QdrantClientService


def get_qdrant(request: Request) -> QdrantClientService:
    return request.app.state.qdrant


def get_gemini(request: Request) -> GeminiClientService:
    return request.app.state.gemini


def get_consumer(request: Request) -> BaseConsumer:
    return request.app.state.consumers[0]
