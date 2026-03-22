from fastapi import Request
from spacy.language import Language

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
