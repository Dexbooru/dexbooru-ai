import numpy as np
from google import genai
from google.genai import types

from models.application.posts import DexbooruPost
from utils.config import get_settings
from utils.image_preprecessor import ImagePreprocessor
from utils.logger import get_logger

logger = get_logger(__name__)


class GeminiClientService(genai.Client):
    AUTO_NORMALIZED_EMBEDDING_DIMENSIONS: int = 3072

    def __init__(self) -> None:
        settings = get_settings()

        super().__init__(
            api_key=settings.gemini_api_key,
        )
        self.embedding_model_name = settings.gemini_embedding_model_name
        self.output_dimensions = settings.gemini_output_dimensions

    def _build_embedding_model_config(self) -> types.EmbedContentConfig:
        return types.EmbedContentConfig(
            output_dimensionality=self.output_dimensions,
        )

    def _noramlize_image_embedding(self, embedding: list[float]) -> list[float]:
        total_dims = len(embedding)
        if total_dims != self.output_dimensions or total_dims == GeminiClientService.AUTO_NORMALIZED_EMBEDDING_DIMENSIONS:
            return embedding

        embedding_np = np.array(embedding, dtype=np.float32)
        embedding_np_normalized = embedding_np / np.linalg.norm(embedding_np)

        return embedding_np_normalized.tolist()

    def is_healthy(self) -> bool:
        models_list_response = self.models.list(config=types.ListModelsConfig(page_size=1))
        return models_list_response.sdk_http_response.headers is not None and models_list_response.config is not None

    def embed_images(self, post: DexbooruPost, image_bytes_list: list[bytes]) -> list[list[float]]:
        if self.embedding_model_name is None:
            raise ValueError("Embedding model name is not set")

        if not image_bytes_list:
            return []

        logger.info(
            "Embedding %s images with model: %s for post: %s",
            len(image_bytes_list),
            self.embedding_model_name,
            post.id,
        )
        contents = [
            types.Part.from_bytes(
                data=img_bytes,
                mime_type=f"image/{ImagePreprocessor.TARGET_IMAGE_MIME_TYPE}",
            )
            for img_bytes in image_bytes_list
        ]

        image_embedding_response = self.models.embed_content(
            model=self.embedding_model_name,
            contents=contents,
            config=self._build_embedding_model_config(),
        )
        logger.info(
            "Embedded %s images for post: %s",
            len(image_embedding_response.embeddings),
            post.id,
        )

        return [self._noramlize_image_embedding(embedding.values) for embedding in image_embedding_response.embeddings]
