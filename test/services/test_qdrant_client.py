import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams

from models.application.posts import DexbooruPost
from services.qdrant_client import QdrantClientService


def _make_service_with_mock_http(
    host: str = "http://localhost:6333/",
) -> QdrantClientService:
    with patch("services.qdrant_client.get_settings") as mock_get_settings:
        mock_get_settings.return_value = MagicMock(
            qdrant_url="http://localhost:6333",
            qdrant_api_key=None,
            environment="development",
            gemini_output_dimensions=1536,
        )
        with patch.object(AsyncQdrantClient, "__init__", lambda self, **kwargs: None):
            svc = QdrantClientService()
    svc._client = MagicMock()
    svc._client.http.client.host = host
    return svc


class TestQdrantClientServiceInit:
    """Tests for QdrantClientService.__init__."""

    def test_reads_settings_and_sets_output_dimensions(self) -> None:
        with patch("services.qdrant_client.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.qdrant_url = "http://localhost:6333"
            mock_settings.qdrant_api_key = None
            mock_settings.environment = "development"
            mock_settings.gemini_output_dimensions = 768
            mock_get_settings.return_value = mock_settings

            with patch.object(AsyncQdrantClient, "__init__", lambda self, **kwargs: None):
                svc = QdrantClientService()

        assert svc.output_dimensions == 768


class TestQdrantClientServiceGetClientBaseUrl:
    """Tests for QdrantClientService._get_client_base_url."""

    def test_returns_host_without_trailing_slash(self) -> None:
        svc = _make_service_with_mock_http("http://localhost:6333/")
        assert svc._get_client_base_url() == "http://localhost:6333"

        svc._client.http.client.host = "http://qdrant:6333"
        assert svc._get_client_base_url() == "http://qdrant:6333"


class TestQdrantClientServiceIsHealthy:
    """Tests for QdrantClientService.is_healthy."""

    def test_returns_true_when_healthz_200_and_expected_text(self) -> None:
        svc = _make_service_with_mock_http("http://localhost:6333")

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value=QdrantClientService.HEALTH_CHECK_PASSED_TEXT)

        with patch("services.qdrant_client.aiohttp.ClientSession") as mock_session:
            cm = MagicMock()
            cm.__aenter__ = AsyncMock(return_value=mock_response)
            cm.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value.__aenter__.return_value.get = MagicMock(return_value=cm)

            result = asyncio.run(svc.is_healthy())

        assert result is True

    def test_returns_false_when_healthz_non_200(self) -> None:
        svc = _make_service_with_mock_http("http://localhost:6333")

        mock_response = AsyncMock()
        mock_response.status = 503
        mock_response.text = AsyncMock(return_value="unavailable")

        with patch("services.qdrant_client.aiohttp.ClientSession") as mock_session:
            cm = MagicMock()
            cm.__aenter__ = AsyncMock(return_value=mock_response)
            cm.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value.__aenter__.return_value.get = MagicMock(return_value=cm)

            result = asyncio.run(svc.is_healthy())

        assert result is False

    def test_returns_false_when_response_text_mismatch(self) -> None:
        svc = _make_service_with_mock_http("http://localhost:6333")

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="something else")

        with patch("services.qdrant_client.aiohttp.ClientSession") as mock_session:
            cm = MagicMock()
            cm.__aenter__ = AsyncMock(return_value=mock_response)
            cm.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value.__aenter__.return_value.get = MagicMock(return_value=cm)

            result = asyncio.run(svc.is_healthy())

        assert result is False

    def test_raises_on_network_error(self) -> None:
        svc = _make_service_with_mock_http("http://localhost:6333")

        with patch("services.qdrant_client.aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__.return_value.get = MagicMock(side_effect=OSError("connection refused"))

            with pytest.raises(OSError):
                asyncio.run(svc.is_healthy())


class TestQdrantClientServiceBuildVectorsConfig:
    """Tests for QdrantClientService._build_vectors_config."""

    def test_returns_vector_params_with_size_and_distance(self) -> None:
        svc = _make_service_with_mock_http()
        svc.output_dimensions = 512
        config = svc._build_vectors_config()
        assert isinstance(config, VectorParams)
        assert config.size == 512
        assert config.distance == QdrantClientService.SIMILARITY_METRIC
        assert config.distance == Distance.COSINE


class TestQdrantClientServiceDoesCollectionExist:
    """Tests for QdrantClientService._does_collection_exist."""

    def test_returns_result_of_collection_exists(self) -> None:
        svc = _make_service_with_mock_http()
        svc.collection_exists = AsyncMock(return_value=True)

        result = asyncio.run(svc._does_collection_exist("my_coll"))

        assert result is True
        svc.collection_exists.assert_awaited_once_with(collection_name="my_coll")

    def test_returns_false_when_collection_does_not_exist(self) -> None:
        svc = _make_service_with_mock_http()
        svc.collection_exists = AsyncMock(return_value=False)

        result = asyncio.run(svc._does_collection_exist("missing"))

        assert result is False


class TestQdrantClientServiceCreateCollection:
    """Tests for QdrantClientService._create_collection."""

    def test_creates_collection_when_recreate_false(self) -> None:
        svc = _make_service_with_mock_http()
        svc.create_collection = AsyncMock()
        svc.delete_collection = AsyncMock()
        vectors_config = VectorParams(size=256, distance=Distance.COSINE)

        result = asyncio.run(svc._create_collection("new_coll", vectors_config=vectors_config, recreate=False))

        assert result is True
        svc.create_collection.assert_awaited_once_with(collection_name="new_coll", vectors_config=vectors_config)
        svc.delete_collection.assert_not_called()

    def test_deletes_and_returns_when_recreate_true(self) -> None:
        svc = _make_service_with_mock_http()
        svc.create_collection = AsyncMock()
        svc.delete_collection = AsyncMock(return_value=True)
        vectors_config = VectorParams(size=256, distance=Distance.COSINE)

        result = asyncio.run(svc._create_collection("old_coll", vectors_config=vectors_config, recreate=True))

        assert result is True
        svc.delete_collection.assert_awaited_once_with(collection_name="old_coll")
        svc.create_collection.assert_not_called()


class TestQdrantClientServiceCreateBaseCollections:
    """Tests for QdrantClientService.create_base_collections."""

    def test_skips_existing_collections(self) -> None:
        svc = _make_service_with_mock_http()
        svc._does_collection_exist = AsyncMock(return_value=True)
        svc._create_collection = AsyncMock()

        result = asyncio.run(svc.create_base_collections())

        assert result is True
        assert svc._does_collection_exist.await_count == len(QdrantClientService.BASE_COLLECTION_NAMES)
        svc._create_collection.assert_not_called()

    def test_creates_missing_collections(self) -> None:
        svc = _make_service_with_mock_http()
        svc._does_collection_exist = AsyncMock(return_value=False)
        svc._create_collection = AsyncMock(return_value=True)

        result = asyncio.run(svc.create_base_collections())

        assert result is True
        assert svc._create_collection.await_count == len(QdrantClientService.BASE_COLLECTION_NAMES)

    def test_returns_false_when_create_fails(self) -> None:
        svc = _make_service_with_mock_http()
        svc._does_collection_exist = AsyncMock(return_value=False)
        svc._create_collection = AsyncMock(return_value=False)

        result = asyncio.run(svc.create_base_collections())

        assert result is False


def _make_post(
    description: str = "test",
    image_urls: list[str] | None = None,
) -> DexbooruPost:
    now = datetime.now(UTC)
    return DexbooruPost(
        id=uuid4(),
        description=description,
        image_urls=image_urls or ["https://cdn.example.com/img1.png"],
        created_at=now,
        updated_at=now,
        author_id=uuid4(),
    )


class TestQdrantClientServiceAddPostImage:
    """Tests for QdrantClientService.add_post_image."""

    def test_upserts_points_and_returns_result(self) -> None:
        svc = _make_service_with_mock_http()
        mock_result = MagicMock()
        mock_result.operation_id = 42
        mock_result.status = MagicMock()
        svc.upsert = AsyncMock(return_value=mock_result)

        post = _make_post()
        embeddings = [[0.1] * 1536, [0.2] * 1536]

        result = asyncio.run(svc.add_post_image(post, embeddings))

        assert result is mock_result
        svc.upsert.assert_awaited_once()
        call_kw = svc.upsert.call_args[1]
        assert call_kw["collection_name"] == QdrantClientService.POST_IMAGE_COLLECTION_NAME
        points = call_kw["points"]
        assert len(points) == 2
        assert points[0].id == post.id
        assert points[0].vector == [0.1] * 1536
        assert points[0].payload == post.model_dump()
        assert points[1].id == post.id
        assert points[1].vector == [0.2] * 1536
        assert points[1].payload == post.model_dump()


class TestQdrantClientServiceSearchPostImageSimilarity:
    """Tests for QdrantClientService.search_post_image_similarity."""

    def test_returns_empty_list_for_empty_query_vector(self) -> None:
        svc = _make_service_with_mock_http()
        svc.query_points = AsyncMock()

        result = asyncio.run(svc.search_post_image_similarity(query_vector=[], limit=5))

        assert result == []
        svc.query_points.assert_not_called()

    def test_normalizes_query_vector_and_maps_results(self) -> None:
        svc = _make_service_with_mock_http()
        mock_hit = MagicMock()
        mock_hit.id = "post-1"
        mock_hit.score = 0.81234
        mock_hit.payload = {"image_urls": ["https://cdn.example.com/1.png"]}
        mock_query_response = MagicMock()
        mock_query_response.points = [mock_hit]
        svc.query_points = AsyncMock(return_value=mock_query_response)

        result = asyncio.run(svc.search_post_image_similarity(query_vector=[3.0, 4.0], limit=2))

        svc.query_points.assert_awaited_once()
        call_kw = svc.query_points.call_args[1]
        assert call_kw["collection_name"] == QdrantClientService.POST_IMAGE_COLLECTION_NAME
        assert call_kw["limit"] == 2
        assert call_kw["with_payload"] is True
        assert call_kw["timeout"] == QdrantClientService.POST_IMAGE_SIMILARITY_QUERY_TIMEOUT_SEC
        assert call_kw["score_threshold"] == QdrantClientService.POST_IMAGE_SIMILARITY_SCORE_THRESHOLD
        assert call_kw["query"] == pytest.approx([0.6, 0.8], rel=1e-6)
        assert result == [
            {
                "post_id": "post-1",
                "image_url": "https://cdn.example.com/1.png",
                "score": 0.81234,
            }
        ]

    def test_keeps_query_vector_when_already_normalized(self) -> None:
        svc = _make_service_with_mock_http()
        mock_query_response = MagicMock()
        mock_query_response.points = []
        svc.query_points = AsyncMock(return_value=mock_query_response)

        asyncio.run(svc.search_post_image_similarity(query_vector=[0.6, 0.8], limit=3))

        call_kw = svc.query_points.call_args[1]
        assert call_kw["query"] == pytest.approx([0.6, 0.8], rel=1e-6)

    def test_maps_empty_image_urls_to_empty_string(self) -> None:
        svc = _make_service_with_mock_http()
        mock_hit = MagicMock()
        mock_hit.id = "post-2"
        mock_hit.score = 0.5
        mock_hit.payload = {}
        mock_query_response = MagicMock()
        mock_query_response.points = [mock_hit]
        svc.query_points = AsyncMock(return_value=mock_query_response)

        result = asyncio.run(svc.search_post_image_similarity(query_vector=[1.0, 0.0], limit=1))

        assert result == [{"post_id": "post-2", "image_url": "", "score": 0.5}]
