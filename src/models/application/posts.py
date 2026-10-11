import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

MAXIMUM_POST_DESCRIPTION_LENGTH = 500


def _collapse_whitespace(value: str) -> str:
    return " ".join(value.split())


def _clean_labels(values: list[str]) -> list[str]:
    return list(dict.fromkeys(label for label in map(_collapse_whitespace, values) if label))


class DexbooruPost(BaseModel):
    id: UUID
    description: str = Field(..., max_length=MAXIMUM_POST_DESCRIPTION_LENGTH)
    image_urls: list[str]
    tags: list[str] = Field(default_factory=list)
    artists: list[str] = Field(default_factory=list)
    created_at: datetime.datetime
    updated_at: datetime.datetime
    author_id: UUID

    @field_validator("description")
    @classmethod
    def _clean_description(cls, description: str) -> str:
        return _collapse_whitespace(description)

    @field_validator("tags")
    @classmethod
    def _clean_tags(cls, tags: list[str]) -> list[str]:
        return _clean_labels([tag.replace("_", " ") for tag in tags])

    @field_validator("artists")
    @classmethod
    def _clean_artists(cls, artists: list[str]) -> list[str]:
        return _clean_labels(artists)

    @staticmethod
    def synthetic_similarity_search_post(description: str | None = None) -> "DexbooruPost":
        now = datetime.datetime.now(datetime.UTC)
        resolved_description = (description or "").strip() or "not provided"
        return DexbooruPost(
            id=uuid4(),
            description=resolved_description,
            image_urls=[],
            created_at=now,
            updated_at=now,
            author_id=uuid4(),
        )
