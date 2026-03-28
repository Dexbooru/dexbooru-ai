import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

MAXIMUM_POST_DESCRIPTION_LENGTH = 500


class DexbooruPost(BaseModel):
    id: UUID
    description: str = Field(..., max_length=MAXIMUM_POST_DESCRIPTION_LENGTH)
    image_urls: list[str]
    created_at: datetime.datetime
    updated_at: datetime.datetime
    author_id: UUID

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
