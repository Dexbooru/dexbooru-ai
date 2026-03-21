import asyncio
from typing import TYPE_CHECKING

from pika.channel import Channel
from pika.spec import Basic, BasicProperties
from qdrant_client.models import UpdateStatus

from core.consumer import BaseConsumer
from models.application.posts import DexbooruPost
from utils import image_preprecessor
from utils.logger import get_logger

if TYPE_CHECKING:
    from services.gemini_client import GeminiClientService
    from services.qdrant_client import QdrantClientService

queue_name = "new_post_vector_target"
routing_key = "new_post.vector_target.#"

logger = get_logger(__name__)


class NewPostConsumer(BaseConsumer):
    qdrant_service: "QdrantClientService | None"
    gemini_service: "GeminiClientService | None"

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
            message_model=DexbooruPost,
            event_loop=event_loop,
            qdrant=qdrant,
            gemini=gemini,
        )
        self.qdrant_service = qdrant
        self.gemini_service = gemini

    async def on_message(
        self,
        channel: Channel,
        method: Basic.Deliver,
        properties: BasicProperties,
        body: DexbooruPost,
    ) -> None:
        if not self.qdrant_service or not self.gemini_service:
            logger.error(
                "Qdrant or Gemini service not initialized in consumer with class name: %s",
                self.__class__.__name__,
            )
            raise RuntimeError("Qdrant or Gemini service not initialized in consumer")

        logger.info("New post event received: %s", body)

        image_transformer = image_preprecessor.ImagePreprocessor(body)
        image_bytes_list = await image_transformer.transform()

        if not image_bytes_list:
            logger.warning("No image bytes produced for post %s", body.id)
            return

        generated_image_embeddings = self.gemini_service.embed_images(body, image_bytes_list)
        logger.info(
            "Generated %s image embeddings for post %s, where each has a size of %s",
            len(generated_image_embeddings),
            body.id,
            len(generated_image_embeddings[0]),
        )

        upsert_result = await self.qdrant_service.add_post_image(body, generated_image_embeddings)
        if upsert_result.status != UpdateStatus.COMPLETED:
            logger.error(
                "Failed to add post image to qdrant for post %s, with operation id %s and status %s",
                body.id,
                upsert_result.operation_id,
                upsert_result.status,
            )
            raise RuntimeError(f"Failed to add post image to qdrant for post {body.id}, with operation id {upsert_result.operation_id} and status {upsert_result.status}")

        logger.info(
            "Successfully added post image to qdrant for post %s, with operation id %s and status %s",
            body.id,
            upsert_result.operation_id,
            upsert_result.status,
        )
