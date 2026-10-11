"""Unit tests for consumers.post_updated_consumer.PostUpdatedConsumer."""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from pika.spec import Basic, BasicProperties

from consumers.post_updated_consumer import PostUpdatedConsumer, queue_name, routing_key
from models.application.posts import DexbooruPost


def _make_post() -> DexbooruPost:
    now = datetime.now(UTC)
    return DexbooruPost(
        id=uuid4(),
        description="edited",
        image_urls=["https://cdn.example.com/img1.png"],
        tags=["cat"],
        artists=["someone"],
        created_at=now,
        updated_at=now,
        author_id=uuid4(),
    )


class TestPostUpdatedConsumerInit:
    """Tests for PostUpdatedConsumer.__init__."""

    def test_sets_queue_name_and_routing_key(self) -> None:
        consumer = PostUpdatedConsumer("amqp://x/", "my_exchange")
        assert consumer.queue_name == queue_name == "post_embedding_sync_updated"
        assert consumer.exchange_name == "my_exchange"
        assert consumer.routing_key == routing_key == "post.embedding_sync.updated"
        assert consumer.message_model is DexbooruPost

    def test_sets_qdrant_and_gemini_services(self) -> None:
        mock_qdrant = MagicMock()
        mock_gemini = MagicMock()
        consumer = PostUpdatedConsumer("amqp://x/", "ex", qdrant=mock_qdrant, gemini=mock_gemini)
        assert consumer.qdrant_service is mock_qdrant
        assert consumer.gemini_service is mock_gemini


class TestPostUpdatedConsumerOnMessage:
    """Tests for PostUpdatedConsumer.on_message."""

    def test_syncs_and_deletes_empty_content(self) -> None:
        consumer = PostUpdatedConsumer("amqp://x/", "ex", qdrant=MagicMock(), gemini=MagicMock())
        consumer.sync_post_embeddings = AsyncMock()
        method = Basic.Deliver(delivery_tag=1, redelivered=False, exchange="ex", routing_key=routing_key)
        body = _make_post()

        asyncio.run(consumer.on_message(MagicMock(), method, BasicProperties(), body))

        consumer.sync_post_embeddings.assert_awaited_once_with(body, delete_when_empty=True)
