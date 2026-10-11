import time

import numpy as np
from google import genai
from google.genai import errors, types

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

    def _normalize_embedding(self, embedding: list[float]) -> list[float]:
        total_dims = len(embedding)
        if total_dims != self.output_dimensions or total_dims == GeminiClientService.AUTO_NORMALIZED_EMBEDDING_DIMENSIONS:
            return embedding

        embedding_np = np.array(embedding, dtype=np.float32)
        embedding_np_normalized = embedding_np / np.linalg.norm(embedding_np)

        return embedding_np_normalized.tolist()

    def _build_post_text_document(self, post: DexbooruPost) -> str:
        return f"title: {post.description} | text: tags: {', '.join(post.tags)}; artists: {', '.join(post.artists)}"

    def _embed_contents(self, post: DexbooruPost, contents: list, content_kind: str) -> list[list[float]]:
        model_name = self.embedding_model_name
        started_at = time.perf_counter()
        try:
            embedding_response = self.models.embed_content(
                model=model_name,
                contents=contents,
                config=self._build_embedding_model_config(),
            )
        except errors.APIError as error:
            raise RuntimeError(f"Gemini {content_kind} embedding request failed for post {post.id} with model {model_name} (status code {error.code}): {error.message}") from error

        embeddings = [embedding.values for embedding in embedding_response.embeddings or [] if embedding.values]
        if not embeddings:
            raise ValueError(f"Gemini returned no {content_kind} embeddings for post {post.id} with model {model_name}")

        has_mismatched_dims = any(len(embedding) != self.output_dimensions for embedding in embeddings)
        if has_mismatched_dims:
            raise ValueError(f"Gemini returned {content_kind} embeddings with dimensions {sorted({len(embedding) for embedding in embeddings})} for post {post.id}, expected {self.output_dimensions}")

        logger.info(
            "%s embedded post_id=%s model=%s embeddings=%s dims=%s duration_ms=%.1f",
            content_kind,
            post.id,
            model_name,
            len(embeddings),
            self.output_dimensions,
            (time.perf_counter() - started_at) * 1000,
        )
        return [self._normalize_embedding(embedding) for embedding in embeddings]

    def is_healthy(self) -> bool:
        models_list_response = self.models.list(config=types.ListModelsConfig(page_size=1))
        return models_list_response.sdk_http_response.headers is not None and models_list_response.config is not None

    def embed_post_text(self, post: DexbooruPost) -> list[float]:
        document = self._build_post_text_document(post)
        logger.debug(
            "embedding post text post_id=%s document_chars=%s tags=%s artists=%s",
            post.id,
            len(document),
            len(post.tags),
            len(post.artists),
        )
        return self._embed_contents(post, [document], "text")[0]

    def embed_images(self, post: DexbooruPost, image_bytes_list: list[bytes]) -> list[list[float]]:
        logger.debug("embedding images post_id=%s images=%s", post.id, len(image_bytes_list))
        contents = [
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=f"image/{ImagePreprocessor.TARGET_IMAGE_MIME_TYPE}",
            )
            for image_bytes in image_bytes_list
        ]
        return self._embed_contents(post, contents, "image")
