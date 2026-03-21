import datetime
from uuid import UUID

from pydantic import BaseModel, Field

MAXIMUM_POST_DESCRIPTION_LENGTH = 500


class DexbooruPost(BaseModel):
    id: UUID
    description: str = Field(..., max_length=MAXIMUM_POST_DESCRIPTION_LENGTH)
    image_urls: list[str]
    created_at: datetime.datetime
    updated_at: datetime.datetime
    author_id: UUID
