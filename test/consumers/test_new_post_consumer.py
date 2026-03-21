"""Unit tests for consumers.new_post_consumer.NewPostConsumer."""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from pika.spec import Basic, BasicProperties
from qdrant_client.models import UpdateStatus

from consumers.new_post_consumer import NewPostConsumer, queue_name, routing_key
from models.application.posts import DexbooruPost


def _make_post(
    post_id: UUID | None = None,
    description: str = "test",
    image_urls: list[str] | None = None,
) -> DexbooruPost:
    uid = uuid4()
    now = datetime.now(UTC)
    return DexbooruPost(
        id=post_id if post_id is not None else uid,
        description=description,
        image_urls=image_urls or ["https://cdn.example.com/img1.png"],
        created_at=now,
        updated_at=now,
        author_id=uuid4(),
    )


class TestNewPostConsumerInit:
    """Tests for NewPostConsumer.__init__."""

    def test_sets_queue_name_and_routing_key(self) -> None:
        consumer = NewPostConsumer("amqp://x/", "my_exchange")
        assert consumer.queue_name == queue_name
        assert consumer.exchange_name == "my_exchange"
        assert consumer.routing_key == routing_key
        assert consumer.message_model is DexbooruPost

    def test_sets_qdrant_and_gemini_services(self) -> None:
        mock_qdrant = MagicMock()
        mock_gemini = MagicMock()
        consumer = NewPostConsumer("amqp://x/", "ex", qdrant=mock_qdrant, gemini=mock_gemini)
        assert consumer.qdrant_service is mock_qdrant
        assert consumer.gemini_service is mock_gemini

    def test_accepts_none_services(self) -> None:
        consumer = NewPostConsumer("amqp://x/", "ex", qdrant=None, gemini=None)
        assert consumer.qdrant_service is None
        assert consumer.gemini_service is None


class TestNewPostConsumerOnMessage:
    """Tests for NewPostConsumer.on_message."""

    def test_raises_when_qdrant_not_initialized(self) -> None:
        consumer = NewPostConsumer("amqp://x/", "ex", qdrant=None, gemini=MagicMock())
        channel = MagicMock()
        method = Basic.Deliver(delivery_tag=1, redelivered=False, exchange="ex", routing_key="rk")
        properties = BasicProperties()
        body = _make_post()

        with pytest.raises(RuntimeError) as exc_info:
            asyncio.run(consumer.on_message(channel, method, properties, body))

        assert "not initialized" in str(exc_info.value)

    def test_raises_when_gemini_not_initialized(self) -> None:
        consumer = NewPostConsumer("amqp://x/", "ex", qdrant=MagicMock(), gemini=None)
        channel = MagicMock()
        method = Basic.Deliver(delivery_tag=1, redelivered=False, exchange="ex", routing_key="rk")
        properties = BasicProperties()
        body = _make_post()

        with pytest.raises(RuntimeError) as exc_info:
            asyncio.run(consumer.on_message(channel, method, properties, body))

        assert "not initialized" in str(exc_info.value)

    def test_success_flow_embeds_and_upserts(self) -> None:
        mock_qdrant = MagicMock()
        mock_gemini = MagicMock()
        mock_qdrant.add_post_image = AsyncMock(
            return_value=MagicMock(
                status=UpdateStatus.COMPLETED,
                operation_id=999,
            )
        )
        mock_gemini.embed_images = MagicMock(return_value=[[0.1] * 256, [0.2] * 256])

        consumer = NewPostConsumer("amqp://x/", "ex", qdrant=mock_qdrant, gemini=mock_gemini)
        channel = MagicMock()
        method = Basic.Deliver(delivery_tag=1, redelivered=False, exchange="ex", routing_key="rk")
        properties = BasicProperties()
        body = _make_post(image_urls=[])

        with (
            patch("consumers.new_post_consumer.image_preprecessor.ImagePreprocessor") as mock_preprocessor_cls,
        ):
            mock_preprocessor = MagicMock()
            mock_preprocessor.transform = AsyncMock(return_value=[b"img1", b"img2"])
            mock_preprocessor_cls.return_value = mock_preprocessor

            asyncio.run(consumer.on_message(channel, method, properties, body))

        mock_preprocessor_cls.assert_called_once_with(body)
        mock_preprocessor.transform.assert_awaited_once()
        mock_gemini.embed_images.assert_called_once_with(body, [b"img1", b"img2"])
        mock_qdrant.add_post_image.assert_awaited_once_with(body, [[0.1] * 256, [0.2] * 256])

    def test_raises_when_upsert_status_not_completed(self) -> None:
        mock_qdrant = MagicMock()
        mock_qdrant.add_post_image = AsyncMock(
            return_value=MagicMock(
                status=UpdateStatus.ACKNOWLEDGED,
                operation_id=111,
            )
        )
        mock_gemini = MagicMock()
        mock_gemini.embed_images = MagicMock(return_value=[[0.0] * 256])

        consumer = NewPostConsumer("amqp://x/", "ex", qdrant=mock_qdrant, gemini=mock_gemini)
        channel = MagicMock()
        method = Basic.Deliver(delivery_tag=1, redelivered=False, exchange="ex", routing_key="rk")
        properties = BasicProperties()
        body = _make_post(image_urls=[])

        with (
            patch("consumers.new_post_consumer.image_preprecessor.ImagePreprocessor") as mock_preprocessor_cls,
        ):
            mock_preprocessor = MagicMock()
            mock_preprocessor.transform = AsyncMock(return_value=[b"img"])
            mock_preprocessor_cls.return_value = mock_preprocessor

            with pytest.raises(RuntimeError) as exc_info:
                asyncio.run(consumer.on_message(channel, method, properties, body))

        assert "Failed to add post image to qdrant" in str(exc_info.value)
        assert "111" in str(exc_info.value)
