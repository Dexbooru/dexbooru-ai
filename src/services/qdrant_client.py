import aiohttp
import numpy as np
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from models.api_responses.post_image_similarity import PostImageSimilarityVectorResult
from models.application.posts import DexbooruPost
from utils.config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


class QdrantClientService(AsyncQdrantClient):
    POST_IMAGE_COLLECTION_NAME: str = "post_image"
    BASE_COLLECTION_NAMES: list[str] = [POST_IMAGE_COLLECTION_NAME]

    SIMILARITY_METRIC: Distance = Distance.COSINE

    POST_IMAGE_SIMILARITY_QUERY_TIMEOUT_SEC: int = 10
    POST_IMAGE_SIMILARITY_SCORE_THRESHOLD: float = 0.5

    HEALTH_CHECK_PASSED_TEXT: str = "healthz check passed"

    def __init__(self) -> None:
        settings = get_settings()

        use_https = settings.qdrant_url.startswith("https") and settings.environment == "production"
        port = 443 if use_https else None

        super().__init__(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            port=port,
            https=use_https,
        )
        self.output_dimensions = settings.gemini_output_dimensions

    @property
    def http(self):
        if hasattr(self._client, "http"):
            return self._client.http
        return super().http

    def _get_client_base_url(self) -> str:
        base_url = self.http.client.host
        return base_url.rstrip("/")

    def _build_vectors_config(self) -> VectorParams:
        return VectorParams(size=self.output_dimensions, distance=self.SIMILARITY_METRIC)

    @staticmethod
    def _normalize_query_vector_for_cosine_search(query_vector: list[float]) -> list[float]:
        vector_np = np.array(query_vector, dtype=np.float32)
        vector_norm = float(np.linalg.norm(vector_np))
        if vector_norm > 0 and not np.isclose(vector_norm, 1.0, atol=1e-3):
            vector_np = vector_np / vector_norm
        return vector_np.tolist()

    @staticmethod
    def _map_scored_points_to_similarity_vector_results(points) -> list[PostImageSimilarityVectorResult]:
        mapped: list[PostImageSimilarityVectorResult] = []
        for result in points or []:
            payload = result.payload or {}
            image_urls = payload.get("image_urls") or []
            image_url = image_urls[0] if image_urls else ""
            mapped.append(
                PostImageSimilarityVectorResult(
                    post_id=str(result.id),
                    image_url=image_url,
                    score=float(result.score),
                )
            )
        return mapped

    async def _does_collection_exist(self, collection_name: str) -> bool:
        return await self.collection_exists(collection_name=collection_name)

    async def _create_collection(self, collection_name: str, vectors_config: VectorParams, recreate: bool = False) -> bool:
        created_collection_successfully = False
        if recreate:
            created_collection_successfully = await self.delete_collection(collection_name=collection_name)
        else:
            await self.create_collection(collection_name=collection_name, vectors_config=vectors_config)
            created_collection_successfully = True
        return created_collection_successfully

    async def create_base_collections(self) -> bool:
        for collection_name in self.BASE_COLLECTION_NAMES:
            collection_exists = await self._does_collection_exist(collection_name)
            if collection_exists:
                logger.info("Collection %s already exists", collection_name)
                continue

            logger.info("Creating collection %s", collection_name)
            base_collections_created_successfully = await self._create_collection(
                collection_name=collection_name,
                vectors_config=self._build_vectors_config(),
                recreate=False,
            )

            if not base_collections_created_successfully:
                logger.error("Failed to create collection %s", collection_name)
                return False

            logger.info("Collection %s created successfully", collection_name)

        return True

    async def add_post_image(self, post: DexbooruPost, image_embeddings: list[list[float]]):
        upsert_result = await self.upsert(
            collection_name=QdrantClientService.POST_IMAGE_COLLECTION_NAME,
            points=[PointStruct(id=post.id, vector=image_embedding, payload=post.model_dump()) for image_embedding in image_embeddings],
        )

        logger.info(
            "Upsert result added for post %s, with operation id %s and status %s",
            post.id,
            upsert_result.operation_id,
            upsert_result.status,
        )
        return upsert_result

    async def search_post_image_similarity(self, query_vector: list[float], limit: int) -> list[PostImageSimilarityVectorResult]:
        if not query_vector:
            return []

        normalized_query = self._normalize_query_vector_for_cosine_search(query_vector)

        search_result = await self.query_points(
            collection_name=QdrantClientService.POST_IMAGE_COLLECTION_NAME,
            query=normalized_query,
            limit=limit,
            with_payload=True,
            timeout=QdrantClientService.POST_IMAGE_SIMILARITY_QUERY_TIMEOUT_SEC,
            score_threshold=QdrantClientService.POST_IMAGE_SIMILARITY_SCORE_THRESHOLD,
        )

        return self._map_scored_points_to_similarity_vector_results(search_result.points)

    async def is_healthy(self) -> bool:
        healthz_url = f"{self._get_client_base_url()}/healthz"
        async with aiohttp.ClientSession() as session:
            async with session.get(healthz_url) as response:
                health_response_text = await response.text()

                return response.status == 200 and health_response_text.strip() == self.HEALTH_CHECK_PASSED_TEXT
