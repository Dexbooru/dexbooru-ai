"""Unit tests for controllers.api.similarity_posts_images_controller."""

from collections.abc import Generator
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from controllers.api import similarity_posts_images_controller
from controllers.api.similarity_posts_images_controller import router as similarity_posts_images_router
from models.api_responses.post_image_similarity import PostImageSimilarityVectorResult
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
                PostImageSimilarityVectorResult(
                    post_id="abc-1",
                    image_url="https://cdn.example.com/a.png",
                    score=0.98456,
                )
            ]
        )
        mock_gemini = MagicMock()
        mock_gemini.embed_images.return_value = [[0.1, 0.2, 0.3]]

        with (
            patch_similarity_deps(app, mock_qdrant, mock_gemini),
            patch.object(
                similarity_posts_images_controller.ImagePreprocessor,
                "resize_image_bytes",
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
        embed_post = mock_gemini.embed_images.call_args[0][0]
        assert embed_post.description == "not provided"

    def test_synthetic_post_uses_request_description_when_provided(self) -> None:
        app = _make_app()
        mock_qdrant = MagicMock()
        mock_qdrant.search_post_image_similarity = AsyncMock(return_value=[])
        mock_gemini = MagicMock()
        mock_gemini.embed_images.return_value = [[0.1, 0.2, 0.3]]

        with (
            patch_similarity_deps(app, mock_qdrant, mock_gemini),
            patch.object(
                similarity_posts_images_controller.ImagePreprocessor,
                "resize_image_bytes",
                return_value=b"resized-image",
            ),
        ):
            client = TestClient(app)
            response = client.post(
                "/similarity/posts/images/",
                data={"description": "query caption", "top_closest_match_count": 5},
                files={"image_file": ("img.png", b"raw-image", "image/png")},
            )

        assert response.status_code == 200
        embed_post = mock_gemini.embed_images.call_args[0][0]
        assert embed_post.description == "query caption"

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

    def test_returns_403_when_image_url_host_is_not_allowed(self) -> None:
        app = _make_app()
        mock_qdrant = MagicMock()
        mock_gemini = MagicMock()
        mock_settings = MagicMock(
            allow_all_image_domain_destinations=False,
            allowed_image_domain_destinations=["cdn.example.com", "images.example.org"],
        )

        with (
            patch_similarity_deps(app, mock_qdrant, mock_gemini),
            patch(
                "controllers.api.similarity_posts_images_controller.get_settings",
                return_value=mock_settings,
            ),
        ):
            client = TestClient(app)
            response = client.post(
                "/similarity/posts/images/",
                data={"image_url": "https://evil.example.com/image.png"},
            )

        assert response.status_code == 403
        assert response.json()["detail"] == ("The provided url does not match any allowed host in this list: cdn.example.com, images.example.org")
        mock_gemini.embed_images.assert_not_called()

    def test_accepts_image_url_when_host_is_allowed(self) -> None:
        app = _make_app()
        mock_qdrant = MagicMock()
        mock_qdrant.search_post_image_similarity = AsyncMock(return_value=[])
        mock_gemini = MagicMock()
        mock_gemini.embed_images.return_value = [[0.1, 0.2, 0.3]]
        mock_settings = MagicMock(
            allow_all_image_domain_destinations=False,
            allowed_image_domain_destinations=["cdn.example.com"],
        )

        response_mock = AsyncMock()
        response_mock.status = 200
        response_mock.headers = {"Content-Type": "image/png"}
        response_mock.read = AsyncMock(return_value=b"raw-image")
        response_mock.__aenter__.return_value = response_mock
        response_mock.__aexit__.return_value = None

        session_mock = MagicMock()
        session_mock.get.return_value = response_mock
        session_mock.__aenter__.return_value = session_mock
        session_mock.__aexit__.return_value = None

        with (
            patch_similarity_deps(app, mock_qdrant, mock_gemini),
            patch(
                "controllers.api.similarity_posts_images_controller.get_settings",
                return_value=mock_settings,
            ),
            patch(
                "controllers.api.similarity_posts_images_controller.aiohttp.ClientSession",
                return_value=session_mock,
            ),
            patch.object(
                similarity_posts_images_controller.ImagePreprocessor,
                "resize_image_bytes",
                return_value=b"resized-image",
            ),
        ):
            client = TestClient(app)
            response = client.post(
                "/similarity/posts/images/",
                data={"image_url": "https://cdn.example.com/image.png"},
            )

        assert response.status_code == 200
        assert response.json() == {"results": []}
        session_mock.get.assert_called_once_with("https://cdn.example.com/image.png")

    def test_skips_host_allowlist_when_all_domains_are_allowed(self) -> None:
        app = _make_app()
        mock_qdrant = MagicMock()
        mock_qdrant.search_post_image_similarity = AsyncMock(return_value=[])
        mock_gemini = MagicMock()
        mock_gemini.embed_images.return_value = [[0.1, 0.2, 0.3]]
        mock_settings = MagicMock(
            allow_all_image_domain_destinations=True,
            allowed_image_domain_destinations=["*"],
        )

        response_mock = AsyncMock()
        response_mock.status = 200
        response_mock.headers = {"Content-Type": "image/png"}
        response_mock.read = AsyncMock(return_value=b"raw-image")
        response_mock.__aenter__.return_value = response_mock
        response_mock.__aexit__.return_value = None

        session_mock = MagicMock()
        session_mock.get.return_value = response_mock
        session_mock.__aenter__.return_value = session_mock
        session_mock.__aexit__.return_value = None

        with (
            patch_similarity_deps(app, mock_qdrant, mock_gemini),
            patch(
                "controllers.api.similarity_posts_images_controller.get_settings",
                return_value=mock_settings,
            ),
            patch(
                "controllers.api.similarity_posts_images_controller.aiohttp.ClientSession",
                return_value=session_mock,
            ),
            patch.object(
                similarity_posts_images_controller.ImagePreprocessor,
                "resize_image_bytes",
                return_value=b"resized-image",
            ),
        ):
            client = TestClient(app)
            response = client.post(
                "/similarity/posts/images/",
                data={"image_url": "https://any-host.example/image.png"},
            )

        assert response.status_code == 200
        assert response.json() == {"results": []}
        session_mock.get.assert_called_once_with("https://any-host.example/image.png")
