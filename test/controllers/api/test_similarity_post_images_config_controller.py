"""Unit tests for controllers.api.similarity_posts_images_config_controller."""

from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from controllers.api.similarity_posts_images_config_controller import router as similarity_post_images_config_router


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(similarity_post_images_config_router)
    return app


class TestSimilarityPostImagesConfigEndpoint:
    """Tests for GET /similarity/posts/images/config/."""

    def test_returns_wildcard_configuration_when_all_domains_allowed(self) -> None:
        app = _make_app()
        mock_settings = MagicMock(
            allowed_image_domain_destinations=["*"],
            allow_all_image_domain_destinations=True,
        )
        with patch(
            "controllers.api.similarity_posts_images_config_controller.get_settings",
            return_value=mock_settings,
        ):
            client = TestClient(app)
            response = client.get("/similarity/posts/images/config/")

        assert response.status_code == 200
        assert response.json() == {
            "allowed_image_domains": ["*"],
            "all_image_domains_allowed_in_search": True,
        }

    def test_returns_explicit_allowed_domains_configuration(self) -> None:
        app = _make_app()
        mock_settings = MagicMock(
            allowed_image_domain_destinations=["cdn.example.com", "images.example.org"],
            allow_all_image_domain_destinations=False,
        )
        with patch(
            "controllers.api.similarity_posts_images_config_controller.get_settings",
            return_value=mock_settings,
        ):
            client = TestClient(app)
            response = client.get("/similarity/posts/images/config/")

        assert response.status_code == 200
        assert response.json() == {
            "allowed_image_domains": ["cdn.example.com", "images.example.org"],
            "all_image_domains_allowed_in_search": False,
        }
