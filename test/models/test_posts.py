"""Unit tests for models.application.posts.DexbooruPost."""

from datetime import UTC, datetime
from uuid import uuid4

from models.application.posts import DexbooruPost


def _make_post(description: str = "a cat", tags: list[str] | None = None, artists: list[str] | None = None) -> DexbooruPost:
    now = datetime.now(UTC)
    return DexbooruPost(
        id=uuid4(),
        description=description,
        image_urls=[],
        tags=tags or [],
        artists=artists or [],
        created_at=now,
        updated_at=now,
        author_id=uuid4(),
    )


class TestDexbooruPostCleaning:
    def test_collapses_description_whitespace(self) -> None:
        assert _make_post(description="  many   spaces\n here ").description == "many spaces here"

    def test_cleans_tags_and_keeps_first_occurrence_order(self) -> None:
        post = _make_post(tags=["cat", " ", "cat", "cat_ears", "cat ears", "long__hair"])

        assert post.tags == ["cat", "cat ears", "long hair"]

    def test_cleans_artists_without_replacing_underscores(self) -> None:
        post = _make_post(artists=["", "bob", "bob ", "jane_doe"])

        assert post.artists == ["bob", "jane_doe"]
