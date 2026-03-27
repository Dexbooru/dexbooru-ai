"""Unit tests for controllers.api.similarity_posts_images_controller."""

from collections.abc import Generator
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from controllers.api import similarity_posts_images_controller
from controllers.api.similarity_posts_images_controller import router as similarity_posts_images_router
from services.gemini_client import GeminiClientService
from services.qdrant_client import QdrantClientService


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(similarity_posts_images_router)
    return app


@contextmanager
def patch_similarity_deps(
    app: FastAPI,
    qdrant: QdrantClientService,
    gemini: GeminiClientService,
) -> Generator[None, None, None]:
    def get_qdrant_override(request: Request) -> QdrantClientService:
        return qdrant

    def get_gemini_override(request: Request) -> GeminiClientService:
        return gemini

    app.dependency_overrides[similarity_posts_images_controller.get_qdrant] = get_qdrant_override
    app.dependency_overrides[similarity_posts_images_controller.get_gemini] = get_gemini_override
    try:
        yield
    finally:
        app.dependency_overrides.clear()


class TestSimilarityPostsImagesEndpoint:
    """Tests for POST /similarity/posts/images/."""

    def test_returns_400_when_both_image_url_and_file_are_provided(self) -> None:
        app = _make_app()
        mock_qdrant = MagicMock()
        mock_gemini = MagicMock()
        with patch_similarity_deps(app, mock_qdrant, mock_gemini):
            client = TestClient(app)
            response = client.post(
                "/similarity/posts/images/",
                data={"image_url": "https://example.com/image.png"},
                files={"image_file": ("image.png", b"pngbytes", "image/png")},
            )

        assert response.status_code == 400
        assert response.json()["detail"] == "Provide exactly one of image_url or image_file"

    def test_returns_400_when_neither_image_url_nor_file_is_provided(self) -> None:
        app = _make_app()
        mock_qdrant = MagicMock()
        mock_gemini = MagicMock()
        with patch_similarity_deps(app, mock_qdrant, mock_gemini):
            client = TestClient(app)
            response = client.post("/similarity/posts/images/")

        assert response.status_code == 400
        assert response.json()["detail"] == "Provide exactly one of image_url or image_file"

    def test_returns_400_when_uploaded_file_exceeds_max_size(self) -> None:
        app = _make_app()
        mock_qdrant = MagicMock()
        mock_gemini = MagicMock()
        payload = b"a" * ((3 * 1024 * 1024) + 1)
        with patch_similarity_deps(app, mock_qdrant, mock_gemini):
            client = TestClient(app)
            response = client.post(
                "/similarity/posts/images/",
                files={"image_file": ("large.png", payload, "image/png")},
            )

        assert response.status_code == 400
        assert response.json()["detail"] == "Uploaded file exceeds 3MB limit"

    def test_returns_results_with_expected_shape_and_score_format(self) -> None:
        app = _make_app()
        mock_qdrant = MagicMock()
        mock_qdrant.search_post_image_similarity = AsyncMock(
            return_value=[
                {
                    "post_id": "abc-1",
                    "image_url": "https://cdn.example.com/a.png",
                    "score": 0.98456,
                }
            ]
        )
        mock_gemini = MagicMock()
        mock_gemini.embed_images.return_value = [[0.1, 0.2, 0.3]]

        with (
            patch_similarity_deps(app, mock_qdrant, mock_gemini),
            patch.object(
                similarity_posts_images_controller.ImagePreprocessor,
                "_resize_to_dimensions",
                return_value=b"resized-image",
            ),
        ):
            client = TestClient(app)
            response = client.post(
                "/similarity/posts/images/",
                data={"top_closest_match_count": 5},
                files={"image_file": ("img.png", b"raw-image", "image/png")},
            )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert data["results"] == [
            {
                "post_id": "abc-1",
                "image_url": "https://cdn.example.com/a.png",
                "similarity_score": 98.46,
            }
        ]
        mock_qdrant.search_post_image_similarity.assert_awaited_once_with(
            query_vector=[0.1, 0.2, 0.3],
            limit=5,
        )

    def test_returns_400_for_unsupported_upload_mime_type(self) -> None:
        app = _make_app()
        mock_qdrant = MagicMock()
        mock_gemini = MagicMock()
        with patch_similarity_deps(app, mock_qdrant, mock_gemini):
            client = TestClient(app)
            response = client.post(
                "/similarity/posts/images/",
                files={"image_file": ("notes.txt", b"hello", "text/plain")},
            )

        assert response.status_code == 400
        assert response.json()["detail"] == "Unsupported uploaded image MIME type"
