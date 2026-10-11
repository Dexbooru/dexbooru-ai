import asyncio
import time
from enum import StrEnum
from uuid import UUID

from qdrant_client.models import UpdateResult, UpdateStatus

from core.consumer import BaseConsumer
from models.application.posts import DexbooruPost
from services.gemini_client import GeminiClientService
from services.qdrant_client import QdrantClientService
from utils import image_preprecessor
from utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingSyncOutcome(StrEnum):
    UPSERTED = "upserted"
    DELETED = "deleted"
    SKIPPED = "skipped"


class BasePostEmbeddingSyncConsumer(BaseConsumer):
    EVENT_NAME: str

    def __init__(
        self,
        amqp_url: str,
        queue_name: str,
        exchange_name: str,
        *,
        routing_key: str,
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
    def _raise_if_write_failed(operation: str, collection_name: str, post_id: UUID, result: UpdateResult) -> None:
        if result.status != UpdateStatus.COMPLETED:
            raise RuntimeError(f"Qdrant {operation} on collection {collection_name} failed for post {post_id}, with operation id {result.operation_id} and status {result.status}")

    async def _sync_post_text(self, post: DexbooruPost, *, delete_when_empty: bool) -> EmbeddingSyncOutcome:
        qdrant, gemini = self.qdrant_service, self.gemini_service
        collection_name = qdrant.POST_TEXT_COLLECTION_NAME

        text_embedding = gemini.embed_post_text(post)
        if text_embedding is None:
            if not delete_when_empty:
                logger.warning("post has no text content, skipping text vector post_id=%s", post.id)
                return EmbeddingSyncOutcome.SKIPPED

            logger.warning("post has no text content, deleting text vector post_id=%s", post.id)
            self._raise_if_write_failed("delete", collection_name, post.id, await qdrant.delete_post_text(post.id))
            return EmbeddingSyncOutcome.DELETED

        self._raise_if_write_failed("upsert", collection_name, post.id, await qdrant.add_post_text(post, text_embedding))
        return EmbeddingSyncOutcome.UPSERTED

    async def _sync_post_images(self, post: DexbooruPost, *, delete_when_empty: bool) -> tuple[EmbeddingSyncOutcome, int]:
        qdrant, gemini = self.qdrant_service, self.gemini_service
        collection_name = qdrant.POST_IMAGE_COLLECTION_NAME

        image_transformer = image_preprecessor.ImagePreprocessor(post)
        cdn_image_urls = image_transformer.cdn_image_urls()
        if not cdn_image_urls:
            if not delete_when_empty:
                logger.warning(
                    "post has no CDN images, skipping image vector post_id=%s image_urls=%s",
                    post.id,
                    len(post.image_urls),
                )
                return EmbeddingSyncOutcome.SKIPPED, 0

            logger.warning(
                "post has no CDN images, deleting image vector post_id=%s image_urls=%s",
                post.id,
                len(post.image_urls),
            )
            self._raise_if_write_failed("delete", collection_name, post.id, await qdrant.delete_post_image(post.id))
            return EmbeddingSyncOutcome.DELETED, 0

        image_bytes_list = await image_transformer.transform()
        image_embeddings = gemini.embed_images(post, image_bytes_list)
        if not image_embeddings:
            raise RuntimeError(f"No usable images for post {post.id}: none of its {len(cdn_image_urls)} CDN image(s) could be downloaded or decoded")

        self._raise_if_write_failed("upsert", collection_name, post.id, await qdrant.add_post_image(post, image_embeddings))
        return EmbeddingSyncOutcome.UPSERTED, len(image_bytes_list)

    async def sync_post_embeddings(self, post: DexbooruPost, *, delete_when_empty: bool) -> None:
        started_at = time.perf_counter()
        logger.info(
            "post embedding sync started post_id=%s event=%s description_chars=%s tags=%s artists=%s image_urls=%s",
            post.id,
            self.EVENT_NAME,
            len(post.description),
            len(post.tags),
            len(post.artists),
            len(post.image_urls),
        )

        text_outcome = await self._sync_post_text(post, delete_when_empty=delete_when_empty)
        image_outcome, images_used = await self._sync_post_images(post, delete_when_empty=delete_when_empty)

        logger.info(
            "post embedding sync completed post_id=%s event=%s text=%s images=%s images_used=%s/%s duration_ms=%.1f",
            post.id,
            self.EVENT_NAME,
            text_outcome,
            image_outcome,
            images_used,
            len(post.image_urls),
            (time.perf_counter() - started_at) * 1000,
        )
