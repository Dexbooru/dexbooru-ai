from fastapi import Request
from pydantic_ai.models.openai import OpenAIChatModel
from redis.asyncio import Redis
from spacy.language import Language

from core.database import Database
from ml.dexbooru_tag_rating_predictor import DexbooruTagRatingPredictor
from services.gemini_client import GeminiClientService
from services.qdrant_client import QdrantClientService


def get_qdrant(request: Request) -> QdrantClientService:
    return request.app.state.qdrant


def get_gemini(request: Request) -> GeminiClientService:
    return request.app.state.gemini


def get_amqp_url(request: Request) -> str:
    return request.app.state.amqp_url


def get_spacy_nlp(request: Request) -> Language:
    return request.app.state.nlp


def get_tag_rating_predictor(request: Request) -> DexbooruTagRatingPredictor:
    return request.app.state.tag_rating_predictor


def get_database(request: Request) -> Database:
    return request.app.state.database


def get_redis(request: Request) -> Redis:
    return request.app.state.redis


def get_chat_model(request: Request) -> OpenAIChatModel:
    return request.app.state.chat_model
