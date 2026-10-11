"""Unit tests for consumers.post_created_consumer.PostCreatedConsumer."""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from pika.spec import Basic, BasicProperties

from consumers.post_created_consumer import PostCreatedConsumer, queue_name, routing_key
from models.application.posts import DexbooruPost


def _make_post() -> DexbooruPost:
    now = datetime.now(UTC)
    return DexbooruPost(
        id=uuid4(),
        description="test",
        image_urls=["https://cdn.example.com/img1.png"],
        created_at=now,
        updated_at=now,
        author_id=uuid4(),
    )


class TestPostCreatedConsumerInit:
    """Tests for PostCreatedConsumer.__init__."""

    def test_sets_queue_name_and_routing_key(self) -> None:
        consumer = PostCreatedConsumer("amqp://x/", "my_exchange", qdrant=MagicMock(), gemini=MagicMock())
        assert consumer.queue_name == queue_name == "post_embedding_sync_created"
        assert consumer.exchange_name == "my_exchange"
        assert consumer.routing_key == routing_key == "post.embedding_sync.created"
        assert consumer.message_model is DexbooruPost

    def test_sets_qdrant_and_gemini_services(self) -> None:
        mock_qdrant = MagicMock()
        mock_gemini = MagicMock()
        consumer = PostCreatedConsumer("amqp://x/", "ex", qdrant=mock_qdrant, gemini=mock_gemini)
        assert consumer.qdrant_service is mock_qdrant
        assert consumer.gemini_service is mock_gemini


class TestPostCreatedConsumerOnMessage:
    """Tests for PostCreatedConsumer.on_message."""

    def test_syncs_without_deleting_empty_content(self) -> None:
        consumer = PostCreatedConsumer("amqp://x/", "ex", qdrant=MagicMock(), gemini=MagicMock())
        consumer.sync_post_embeddings = AsyncMock()
        method = Basic.Deliver(delivery_tag=1, redelivered=False, exchange="ex", routing_key=routing_key)
        body = _make_post()

        asyncio.run(consumer.on_message(MagicMock(), method, BasicProperties(), body))

        consumer.sync_post_embeddings.assert_awaited_once_with(body, delete_when_empty=False)
