import asyncio
from typing import TYPE_CHECKING

from pika.channel import Channel
from pika.spec import Basic, BasicProperties

from consumers.post_embedding_sync_consumer import BasePostEmbeddingSyncConsumer
from models.application.posts import DexbooruPost

if TYPE_CHECKING:
    from services.gemini_client import GeminiClientService
    from services.qdrant_client import QdrantClientService

queue_name = "post_embedding_sync_updated"
routing_key = "post.embedding_sync.updated"


class PostUpdatedConsumer(BasePostEmbeddingSyncConsumer):
    EVENT_NAME = "updated"

    def __init__(
        self,
        amqp_url: str,
        exchange_name: str,
        *,
        event_loop: asyncio.AbstractEventLoop | None = None,
        qdrant: "QdrantClientService | None" = None,
        gemini: "GeminiClientService | None" = None,
    ):
        super().__init__(
            amqp_url,
            queue_name,
            exchange_name,
            routing_key=routing_key,
            event_loop=event_loop,
            qdrant=qdrant,
            gemini=gemini,
        )

    async def on_message(
        self,
        channel: Channel,
        method: Basic.Deliver,
        properties: BasicProperties,
        body: DexbooruPost,
    ) -> None:
        await self.sync_post_embeddings(body, delete_when_empty=True)
