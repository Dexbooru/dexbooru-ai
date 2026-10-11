import time
from collections.abc import Iterable

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
    EMPTY_DOCUMENT_FIELD: str = "none"

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

    def _require_embedding_model_name(self) -> str:
        if self.embedding_model_name is None:
            raise ValueError("Embedding model name is not set")
        return self.embedding_model_name

    def _normalize_embedding(self, embedding: list[float]) -> list[float]:
        total_dims = len(embedding)
        if total_dims != self.output_dimensions or total_dims == GeminiClientService.AUTO_NORMALIZED_EMBEDDING_DIMENSIONS:
            return embedding

        embedding_np = np.array(embedding, dtype=np.float32)
        embedding_np_normalized = embedding_np / np.linalg.norm(embedding_np)

        return embedding_np_normalized.tolist()

    @staticmethod
    def _clean_text(value: str) -> str:
        return " ".join(value.split())

    @staticmethod
    def _clean_terms(values: Iterable[str], *, underscores_to_spaces: bool = False) -> list[str]:
        cleaned_terms: list[str] = []
        seen: set[str] = set()
        for value in values:
            term = value.replace("_", " ") if underscores_to_spaces else value
            term = GeminiClientService._clean_text(term)
            if term and term not in seen:
                seen.add(term)
                cleaned_terms.append(term)
        return cleaned_terms

    @staticmethod
    def build_post_text_document(post: DexbooruPost) -> str | None:
        title = GeminiClientService._clean_text(post.description)
        tags = GeminiClientService._clean_terms(post.tags, underscores_to_spaces=True)
        artists = GeminiClientService._clean_terms(post.artists)

        if not title and not tags and not artists:
            return None

        text_sections = []
        if tags:
            text_sections.append(f"tags: {', '.join(tags)}")
        if artists:
            text_sections.append(f"artists: {', '.join(artists)}")

        resolved_title = title or GeminiClientService.EMPTY_DOCUMENT_FIELD
        resolved_text = "; ".join(text_sections) or GeminiClientService.EMPTY_DOCUMENT_FIELD
        return f"title: {resolved_title} | text: {resolved_text}"

    def _embed_contents(self, post: DexbooruPost, contents: list, content_kind: str) -> list[list[float]]:
        model_name = self._require_embedding_model_name()
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

        mismatched_dims = {len(embedding) for embedding in embeddings if len(embedding) != self.output_dimensions}
        if mismatched_dims:
            raise ValueError(f"Gemini returned {content_kind} embeddings with dimensions {sorted(mismatched_dims)} for post {post.id}, expected {self.output_dimensions}")

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

    def embed_post_text(self, post: DexbooruPost) -> list[float] | None:
        self._require_embedding_model_name()

        document = self.build_post_text_document(post)
        if document is None:
            return None

        logger.debug(
            "embedding post text post_id=%s document_chars=%s tags=%s artists=%s",
            post.id,
            len(document),
            len(post.tags),
            len(post.artists),
        )
        return self._embed_contents(post, [document], "text")[0]

    def embed_images(self, post: DexbooruPost, image_bytes_list: list[bytes]) -> list[list[float]]:
        self._require_embedding_model_name()

        usable_image_bytes = [image_bytes for image_bytes in image_bytes_list if image_bytes]
        if len(usable_image_bytes) != len(image_bytes_list):
            logger.warning(
                "dropped empty image bytes post_id=%s dropped=%s",
                post.id,
                len(image_bytes_list) - len(usable_image_bytes),
            )
        if not usable_image_bytes:
            return []

        logger.debug("embedding images post_id=%s images=%s", post.id, len(usable_image_bytes))
        contents = [
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=f"image/{ImagePreprocessor.TARGET_IMAGE_MIME_TYPE}",
            )
            for image_bytes in usable_image_bytes
        ]
        return self._embed_contents(post, contents, "image")
