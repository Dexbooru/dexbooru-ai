"""Unit tests for consumers.post_embedding_sync_consumer.PostEmbeddingSyncConsumer."""

import asyncio
import logging
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from pika.spec import Basic, BasicProperties
from qdrant_client.models import UpdateStatus

from consumers import CONSUMER_CLASSES
from consumers.post_embedding_sync_consumer import PostEmbeddingSyncConsumer, queue_name, routing_key
from models.application.posts import DexbooruPost

PREPROCESSOR_PATH = "consumers.post_embedding_sync_consumer.image_preprecessor.ImagePreprocessor"
CDN_URL = "https://cdn.example.com/img1.png"
CREATED_ROUTING_KEY = "post.embedding_sync.created"


def _make_post(description: str = "a cat") -> DexbooruPost:
    now = datetime.now(UTC)
    return DexbooruPost(
        id=uuid4(),
        description=description,
        image_urls=[CDN_URL],
        tags=["cat"],
        artists=["someone"],
        created_at=now,
        updated_at=now,
        author_id=uuid4(),
    )


def _write_result(status: UpdateStatus = UpdateStatus.COMPLETED, operation_id: int = 1) -> MagicMock:
    return MagicMock(status=status, operation_id=operation_id)


def _make_qdrant() -> MagicMock:
    qdrant = MagicMock()
    qdrant.add_post_text = AsyncMock(return_value=_write_result())
    qdrant.add_post_image = AsyncMock(return_value=_write_result())
    return qdrant


def _make_gemini() -> MagicMock:
    gemini = MagicMock()
    gemini.embed_post_text = MagicMock(return_value=[0.1] * 4)
    gemini.embed_images = MagicMock(return_value=[[0.2] * 4])
    return gemini


def _make_consumer(qdrant: MagicMock, gemini: MagicMock) -> PostEmbeddingSyncConsumer:
    return PostEmbeddingSyncConsumer("amqp://x/", "ex", qdrant=qdrant, gemini=gemini)


def _consume(consumer: PostEmbeddingSyncConsumer, post: DexbooruPost, image_bytes: list[bytes]) -> None:
    method = Basic.Deliver(delivery_tag=1, redelivered=False, exchange="ex", routing_key=CREATED_ROUTING_KEY)
    with patch(PREPROCESSOR_PATH) as mock_cls:
        mock_cls.return_value.transform = AsyncMock(return_value=image_bytes)
        asyncio.run(consumer.on_message(MagicMock(), method, BasicProperties(), post))


class TestPostEmbeddingSyncConsumerInit:
    def test_binds_one_queue_to_created_and_updated_events(self) -> None:
        qdrant, gemini = _make_qdrant(), _make_gemini()
        consumer = _make_consumer(qdrant, gemini)

        assert consumer.queue_name == queue_name == "post_embedding_sync"
        assert consumer.routing_key == routing_key == "post.embedding_sync.*"
        assert consumer.message_model is DexbooruPost
        assert consumer.qdrant_service is qdrant
        assert consumer.gemini_service is gemini

    def test_is_registered(self) -> None:
        assert PostEmbeddingSyncConsumer in CONSUMER_CLASSES


class TestPostEmbeddingSyncConsumerOnMessage:
    def test_upserts_text_then_images_and_logs_summary(self, caplog: pytest.LogCaptureFixture) -> None:
        qdrant, gemini = _make_qdrant(), _make_gemini()
        post = _make_post()

        with caplog.at_level(logging.INFO, logger="app"):
            _consume(_make_consumer(qdrant, gemini), post, [b"img"])

        gemini.embed_post_text.assert_called_once_with(post)
        qdrant.add_post_text.assert_awaited_once_with(post, [0.1] * 4)
        gemini.embed_images.assert_called_once_with(post, [b"img"])
        qdrant.add_post_image.assert_awaited_once_with(post, [[0.2] * 4])
        summary = [r.getMessage() for r in caplog.records if "post embedding sync completed" in r.getMessage()]
        assert len(summary) == 1
        assert f"post_id={post.id}" in summary[0]
        assert f"routing_key={CREATED_ROUTING_KEY} images_used=1/1" in summary[0]

    def test_does_not_log_post_content(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.DEBUG, logger="app"):
            _consume(_make_consumer(_make_qdrant(), _make_gemini()), _make_post(description="secret description text"), [b"img"])

        assert all("secret description text" not in r.getMessage() for r in caplog.records)

    def test_raises_when_no_images_are_usable(self) -> None:
        qdrant, gemini = _make_qdrant(), _make_gemini()

        with pytest.raises(RuntimeError, match="No usable images"):
            _consume(_make_consumer(qdrant, gemini), _make_post(), [])

        qdrant.add_post_text.assert_awaited_once()
        gemini.embed_images.assert_not_called()
        qdrant.add_post_image.assert_not_called()

    def test_raises_when_text_upsert_not_completed(self) -> None:
        qdrant = _make_qdrant()
        qdrant.add_post_text.return_value = _write_result(UpdateStatus.ACKNOWLEDGED, operation_id=111)

        with pytest.raises(RuntimeError) as exc_info:
            _consume(_make_consumer(qdrant, _make_gemini()), _make_post(), [b"img"])

        assert "post_text" in str(exc_info.value)
        assert "111" in str(exc_info.value)
        qdrant.add_post_image.assert_not_called()

    def test_raises_when_image_upsert_not_completed(self) -> None:
        qdrant = _make_qdrant()
        qdrant.add_post_image.return_value = _write_result(UpdateStatus.ACKNOWLEDGED, operation_id=222)

        with pytest.raises(RuntimeError) as exc_info:
            _consume(_make_consumer(qdrant, _make_gemini()), _make_post(), [b"img"])

        assert "post_image" in str(exc_info.value)
        assert "222" in str(exc_info.value)
