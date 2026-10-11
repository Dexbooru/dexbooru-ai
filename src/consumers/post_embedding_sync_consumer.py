import asyncio
import time
from uuid import UUID

from pika.channel import Channel
from pika.spec import Basic, BasicProperties
from qdrant_client.models import UpdateResult, UpdateStatus

from core.consumer import BaseConsumer
from models.application.posts import DexbooruPost
from services.gemini_client import GeminiClientService
from services.qdrant_client import QdrantClientService
from utils import image_preprecessor
from utils.logger import get_logger

logger = get_logger(__name__)

# One queue for created and updated events keeps them in publish order, so a delayed create can't overwrite a newer edit
queue_name = "post_embedding_sync"
routing_key = "post.embedding_sync.*"


class PostEmbeddingSyncConsumer(BaseConsumer):
    def __init__(
        self,
        amqp_url: str,
        exchange_name: str,
        *,
        qdrant: QdrantClientService,
        gemini: GeminiClientService,
        event_loop: asyncio.AbstractEventLoop | None = None,
    ):
        super().__init__(
            amqp_url,
            queue_name,
            exchange_name,
            routing_key=routing_key,
            message_model=DexbooruPost,
            event_loop=event_loop,
        )
        self.qdrant_service = qdrant
        self.gemini_service = gemini

    @staticmethod
    def _raise_if_write_failed(collection_name: str, post_id: UUID, result: UpdateResult) -> None:
        if result.status != UpdateStatus.COMPLETED:
            raise RuntimeError(f"Qdrant upsert on collection {collection_name} failed for post {post_id}, with operation id {result.operation_id} and status {result.status}")

    async def _sync_post_text(self, post: DexbooruPost) -> None:
        text_embedding = await asyncio.to_thread(self.gemini_service.embed_post_text, post)
        self._raise_if_write_failed(
            QdrantClientService.POST_TEXT_COLLECTION_NAME,
            post.id,
            await self.qdrant_service.add_post_text(post, text_embedding),
        )

    async def _sync_post_images(self, post: DexbooruPost) -> int:
        image_bytes_list = await image_preprecessor.ImagePreprocessor(post).transform()
        if not image_bytes_list:
            raise RuntimeError(f"No usable images for post {post.id}: none of its {len(post.image_urls)} image(s) could be downloaded or decoded")

        image_embeddings = await asyncio.to_thread(self.gemini_service.embed_images, post, image_bytes_list)
        self._raise_if_write_failed(
            QdrantClientService.POST_IMAGE_COLLECTION_NAME,
            post.id,
            await self.qdrant_service.add_post_image(post, image_embeddings),
        )
        return len(image_bytes_list)

    async def on_message(
        self,
        channel: Channel,
        method: Basic.Deliver,
        properties: BasicProperties,
        body: DexbooruPost,
    ) -> None:
        started_at = time.perf_counter()
        logger.info(
            "post embedding sync started post_id=%s routing_key=%s description_chars=%s tags=%s artists=%s image_urls=%s",
            body.id,
            method.routing_key,
            len(body.description),
            len(body.tags),
            len(body.artists),
            len(body.image_urls),
        )

        await self._sync_post_text(body)
        images_used = await self._sync_post_images(body)

        logger.info(
            "post embedding sync completed post_id=%s routing_key=%s images_used=%s/%s duration_ms=%.1f",
            body.id,
            method.routing_key,
            images_used,
            len(body.image_urls),
            (time.perf_counter() - started_at) * 1000,
        )
