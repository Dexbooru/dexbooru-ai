import aiohttp
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams

from utils.config import get_settings


class QdrantClientService:
    BASE_COLLECTION_NAMES: list[str] = ["post_text", "post_image"]
    SIMILARITY_METRIC: Distance = Distance.COSINE
    HEALTH_CHECK_PASSED_TEXT: str = "healthz check passed"

    def __init__(self) -> None:
        settings = get_settings()
        self.client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            https=settings.qdrant_url.startswith("https")
            and settings.environment == "production",
        )
        self.output_dimensions = settings.gemini_output_dimensions

    def _get_client_base_url(self) -> str:
        base_url = self.client.http.client.host
        return base_url.rstrip("/")

    def _build_vectors_config(self) -> VectorParams:
        return VectorParams(
            size=self.output_dimensions, distance=self.SIMILARITY_METRIC
        )

    async def _create_collection(
        self, collection_name: str, vectors_config: VectorParams, recreate: bool = False
    ) -> bool:
        created_collection_successfully = False
        if recreate:
            created_collection_successfully = await self.client.delete_collection(
                collection_name=collection_name
            )
        else:
            created_collection_successfully = await self.client.create_collection(
                collection_name=collection_name, vectors_config=vectors_config
            )
        return created_collection_successfully

    async def create_base_collections(self) -> bool:
        base_collections_created_successfully = False

        for collection_name in self.BASE_COLLECTION_NAMES:
            base_collections_created_successfully = await self._create_collection(
                collection_name=collection_name,
                vectors_config=self._build_vectors_config(),
                recreate=True,
            )

        return base_collections_created_successfully

    async def is_healthy(self) -> bool:
        healthz_url = f"{self._get_client_base_url()}/healthz"
        async with aiohttp.ClientSession() as session:
            async with session.get(healthz_url) as response:
                health_response_text = await response.text()

                return (
                    response.status == 200
                    and health_response_text.strip() == self.HEALTH_CHECK_PASSED_TEXT
                )
