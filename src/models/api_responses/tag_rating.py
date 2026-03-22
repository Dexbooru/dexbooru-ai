from pydantic import BaseModel, Field


class TagRatingPredictionRequest(BaseModel):
    tag_string: str = Field(min_length=1)


class TagRatingPredictionResponse(BaseModel):
    transformed_input: str
    predicted_class: str
    class_probabilities_percent: dict[str, float]
