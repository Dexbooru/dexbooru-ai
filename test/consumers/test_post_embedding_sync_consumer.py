"""Unit tests for consumers.post_embedding_sync_consumer.BasePostEmbeddingSyncConsumer."""

import asyncio
import logging
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from qdrant_client.models import UpdateStatus

from consumers.post_embedding_sync_consumer import BasePostEmbeddingSyncConsumer
from models.application.posts import DexbooruPost
from services.qdrant_client import QdrantClientService

PREPROCESSOR_PATH = "consumers.post_embedding_sync_consumer.image_preprecessor.ImagePreprocessor"
CDN_URL = "https://cdn.example.com/img1.png"


class _SyncConsumer(BasePostEmbeddingSyncConsumer):
    EVENT_NAME = "test"

    def on_message(self, channel, method, properties, body):  # noqa: ANN001
        raise NotImplementedError


def _make_post(
    description: str = "a cat",
    image_urls: list[str] | None = None,
    tags: list[str] | None = None,
    artists: list[str] | None = None,
) -> DexbooruPost:
    now = datetime.now(UTC)
    return DexbooruPost(
        id=uuid4(),
        description=description,
        image_urls=[CDN_URL] if image_urls is None else image_urls,
        tags=tags or [],
        artists=artists or [],
        created_at=now,
        updated_at=now,
        author_id=uuid4(),
    )


def _write_result(status: UpdateStatus = UpdateStatus.COMPLETED, operation_id: int = 1) -> MagicMock:
    return MagicMock(status=status, operation_id=operation_id)


def _make_qdrant() -> MagicMock:
    qdrant = MagicMock()
    qdrant.POST_TEXT_COLLECTION_NAME = QdrantClientService.POST_TEXT_COLLECTION_NAME
    qdrant.POST_IMAGE_COLLECTION_NAME = QdrantClientService.POST_IMAGE_COLLECTION_NAME
    qdrant.add_post_text = AsyncMock(return_value=_write_result())
    qdrant.add_post_image = AsyncMock(return_value=_write_result())
    qdrant.delete_post_text = AsyncMock(return_value=_write_result())
    qdrant.delete_post_image = AsyncMock(return_value=_write_result())
    return qdrant


def _make_gemini(text_embedding: list[float] | None = None, image_embeddings: list[list[float]] | None = None) -> MagicMock:
    gemini = MagicMock()
    gemini.embed_post_text = MagicMock(return_value=[0.1] * 4 if text_embedding is None else text_embedding)
    gemini.embed_images = MagicMock(return_value=[[0.2] * 4] if image_embeddings is None else image_embeddings)
    return gemini


def _mock_preprocessor(mock_cls: MagicMock, cdn_urls: list[str], image_bytes: list[bytes]) -> MagicMock:
    preprocessor = MagicMock()
    preprocessor.cdn_image_urls = MagicMock(return_value=cdn_urls)
    preprocessor.transform = AsyncMock(return_value=image_bytes)
    mock_cls.return_value = preprocessor
    return preprocessor


def _sync(consumer: _SyncConsumer, post: DexbooruPost, *, delete_when_empty: bool) -> None:
    asyncio.run(consumer.sync_post_embeddings(post, delete_when_empty=delete_when_empty))


class TestSyncPostEmbeddingsSuccess:
    def test_upserts_text_then_images_and_logs_summary(self, caplog: pytest.LogCaptureFixture) -> None:
        qdrant, gemini = _make_qdrant(), _make_gemini()
        consumer = _SyncConsumer("amqp://x/", "q", "ex", routing_key="rk", qdrant=qdrant, gemini=gemini)
        post = _make_post(tags=["cat"])

        with patch(PREPROCESSOR_PATH) as mock_cls, caplog.at_level(logging.INFO, logger="app"):
            _mock_preprocessor(mock_cls, [CDN_URL], [b"img"])
            _sync(consumer, post, delete_when_empty=False)

        gemini.embed_post_text.assert_called_once_with(post)
        qdrant.add_post_text.assert_awaited_once_with(post, [0.1] * 4)
        gemini.embed_images.assert_called_once_with(post, [b"img"])
        qdrant.add_post_image.assert_awaited_once_with(post, [[0.2] * 4])
        summary = [r.getMessage() for r in caplog.records if "post embedding sync completed" in r.getMessage()]
        assert len(summary) == 1
        assert f"post_id={post.id}" in summary[0]
        assert "text=upserted images=upserted images_used=1/1" in summary[0]

    def test_summary_does_not_log_post_content(self, caplog: pytest.LogCaptureFixture) -> None:
        consumer = _SyncConsumer("amqp://x/", "q", "ex", routing_key="rk", qdrant=_make_qdrant(), gemini=_make_gemini())
        post = _make_post(description="secret description text")

        with patch(PREPROCESSOR_PATH) as mock_cls, caplog.at_level(logging.DEBUG, logger="app"):
            _mock_preprocessor(mock_cls, [CDN_URL], [b"img"])
            _sync(consumer, post, delete_when_empty=False)

        assert all("secret description text" not in r.getMessage() for r in caplog.records)


class TestSyncPostEmbeddingsEmptyText:
    def test_skips_text_on_create(self, caplog: pytest.LogCaptureFixture) -> None:
        qdrant = _make_qdrant()
        gemini = _make_gemini()
        gemini.embed_post_text.return_value = None
        consumer = _SyncConsumer("amqp://x/", "q", "ex", routing_key="rk", qdrant=qdrant, gemini=gemini)

        with patch(PREPROCESSOR_PATH) as mock_cls, caplog.at_level(logging.WARNING, logger="app"):
            _mock_preprocessor(mock_cls, [CDN_URL], [b"img"])
            _sync(consumer, _make_post(), delete_when_empty=False)

        qdrant.add_post_text.assert_not_called()
        qdrant.delete_post_text.assert_not_called()
        qdrant.add_post_image.assert_awaited_once()
        assert any("skipping text vector" in r.getMessage() for r in caplog.records)

    def test_deletes_text_on_update(self) -> None:
        qdrant = _make_qdrant()
        gemini = _make_gemini()
        gemini.embed_post_text.return_value = None
        consumer = _SyncConsumer("amqp://x/", "q", "ex", routing_key="rk", qdrant=qdrant, gemini=gemini)
        post = _make_post()

        with patch(PREPROCESSOR_PATH) as mock_cls:
            _mock_preprocessor(mock_cls, [CDN_URL], [b"img"])
            _sync(consumer, post, delete_when_empty=True)

        qdrant.delete_post_text.assert_awaited_once_with(post.id)
        qdrant.add_post_text.assert_not_called()


class TestSyncPostEmbeddingsNoImages:
    def test_indexes_text_and_skips_images_on_create(self) -> None:
        qdrant, gemini = _make_qdrant(), _make_gemini()
        consumer = _SyncConsumer("amqp://x/", "q", "ex", routing_key="rk", qdrant=qdrant, gemini=gemini)

        with patch(PREPROCESSOR_PATH) as mock_cls:
            preprocessor = _mock_preprocessor(mock_cls, [], [])
            _sync(consumer, _make_post(image_urls=[]), delete_when_empty=False)

        qdrant.add_post_text.assert_awaited_once()
        preprocessor.transform.assert_not_called()
        qdrant.add_post_image.assert_not_called()
        qdrant.delete_post_image.assert_not_called()

    def test_deletes_image_vector_on_update(self) -> None:
        qdrant, gemini = _make_qdrant(), _make_gemini()
        consumer = _SyncConsumer("amqp://x/", "q", "ex", routing_key="rk", qdrant=qdrant, gemini=gemini)
        post = _make_post(image_urls=["https://elsewhere.example.com/a.png"])

        with patch(PREPROCESSOR_PATH) as mock_cls:
            _mock_preprocessor(mock_cls, [], [])
            _sync(consumer, post, delete_when_empty=True)

        qdrant.delete_post_image.assert_awaited_once_with(post.id)
        gemini.embed_images.assert_not_called()

    @pytest.mark.parametrize("delete_when_empty", [False, True])
    def test_raises_without_deleting_when_all_images_unusable(self, delete_when_empty: bool) -> None:
        qdrant = _make_qdrant()
        gemini = _make_gemini(image_embeddings=[])
        consumer = _SyncConsumer("amqp://x/", "q", "ex", routing_key="rk", qdrant=qdrant, gemini=gemini)

        with patch(PREPROCESSOR_PATH) as mock_cls:
            _mock_preprocessor(mock_cls, [CDN_URL], [])
            with pytest.raises(RuntimeError, match="No usable images"):
                _sync(consumer, _make_post(), delete_when_empty=delete_when_empty)

        qdrant.add_post_text.assert_awaited_once()
        qdrant.delete_post_image.assert_not_called()
        qdrant.add_post_image.assert_not_called()


class TestSyncPostEmbeddingsWriteFailures:
    def test_raises_when_text_upsert_not_completed(self) -> None:
        qdrant = _make_qdrant()
        qdrant.add_post_text.return_value = _write_result(UpdateStatus.ACKNOWLEDGED, operation_id=111)
        consumer = _SyncConsumer("amqp://x/", "q", "ex", routing_key="rk", qdrant=qdrant, gemini=_make_gemini())

        with patch(PREPROCESSOR_PATH) as mock_cls:
            _mock_preprocessor(mock_cls, [CDN_URL], [b"img"])
            with pytest.raises(RuntimeError) as exc_info:
                _sync(consumer, _make_post(), delete_when_empty=False)

        assert "post_text" in str(exc_info.value)
        assert "111" in str(exc_info.value)
        qdrant.add_post_image.assert_not_called()

    def test_raises_when_image_upsert_not_completed(self) -> None:
        qdrant = _make_qdrant()
        qdrant.add_post_image.return_value = _write_result(UpdateStatus.ACKNOWLEDGED, operation_id=222)
        consumer = _SyncConsumer("amqp://x/", "q", "ex", routing_key="rk", qdrant=qdrant, gemini=_make_gemini())

        with patch(PREPROCESSOR_PATH) as mock_cls:
            _mock_preprocessor(mock_cls, [CDN_URL], [b"img"])
            with pytest.raises(RuntimeError) as exc_info:
                _sync(consumer, _make_post(), delete_when_empty=False)

        assert "post_image" in str(exc_info.value)
        assert "222" in str(exc_info.value)
