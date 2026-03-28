from pydantic import BaseModel, Field

from models.application.posts import MAXIMUM_POST_DESCRIPTION_LENGTH


class PostImageSimilaritySearchForm(BaseModel):
    image_url: str | None = None
    top_closest_match_count: int = Field(default=5, ge=1)
    description: str | None = Field(default=None, max_length=MAXIMUM_POST_DESCRIPTION_LENGTH)


class PostImageSimilarityResult(BaseModel):
    post_id: str
    image_url: str
    similarity_score: float


class PostImageSimilarityVectorResult(BaseModel):
    post_id: str
    image_url: str
    score: float


class PostImageSimilaritySearchResponse(BaseModel):
    results: list[PostImageSimilarityResult]
