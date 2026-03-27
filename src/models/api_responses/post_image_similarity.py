from pydantic import BaseModel, Field


class PostImageSimilaritySearchForm(BaseModel):
    image_url: str | None = None
    top_closest_match_count: int = Field(default=5, ge=1)


class PostImageSimilarityResult(BaseModel):
    post_id: str
    image_url: str
    similarity_score: float


class PostImageSimilaritySearchResponse(BaseModel):
    results: list[PostImageSimilarityResult]
