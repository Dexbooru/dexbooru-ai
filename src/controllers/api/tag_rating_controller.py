from fastapi import APIRouter, Depends, HTTPException

from core.dependencies import get_tag_rating_predictor
from ml.dexbooru_tag_rating_predictor import DexbooruTagRatingPredictor
from models.api_responses.tag_rating import TagRatingPredictionRequest, TagRatingPredictionResponse
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/tag-rating", tags=["ml"])


def _truncate(s: str, max_len: int = 100) -> str:
    s = s.strip() or "(empty)"
    return s if len(s) <= max_len else f"{s[:max_len]}..."


@router.post("/predict", response_model=TagRatingPredictionResponse)
async def predict_tag_rating(
    body: TagRatingPredictionRequest,
    predictor: DexbooruTagRatingPredictor = Depends(get_tag_rating_predictor),
) -> TagRatingPredictionResponse:
    tag_preview = _truncate(body.tag_string)
    logger.info("tag-rating predict request | input=%r (len=%d)", tag_preview, len(body.tag_string or ""))

    try:
        transformed, predicted, percents = predictor.predict(body.tag_string)
    except ValueError as e:
        logger.warning("tag-rating predict rejected | input=%r | error=%s", tag_preview, e)
        raise HTTPException(status_code=400, detail=str(e)) from e

    top_probs = dict(sorted(percents.items(), key=lambda x: -x[1])[:2])
    logger.info(
        "tag-rating predict success | input=%r | pred=%s | top_probs=%s",
        tag_preview,
        predicted,
        top_probs,
    )
    return TagRatingPredictionResponse(
        transformed_input=transformed,
        predicted_class=predicted,
        class_probabilities_percent=percents,
    )
