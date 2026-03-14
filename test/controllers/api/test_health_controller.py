"""Unit tests for controllers.api.health_controller."""

from collections.abc import Generator
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from controllers.api import health_controller
from controllers.api.health_controller import router as health_router
from core.consumer import BaseConsumer
from models.api_responses.health import HealthResponse
from services.gemini_client import GeminiClientService
from services.qdrant_client import QdrantClientService


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(health_router)
    return app


class TestHealthCheckEndpoint:
    """Tests for GET /health/."""

    def test_returns_200_when_all_healthy(self) -> None:
        app = _make_app()
        mock_qdrant = MagicMock()
        mock_gemini = MagicMock()
        mock_consumer = MagicMock()
        mock_consumer.amqp_url = "amqp://localhost/"

        mock_qdrant.is_healthy = AsyncMock(return_value=True)
        mock_gemini.is_healthy.return_value = True

        with (
            patch_health_deps(app, mock_qdrant, mock_gemini, mock_consumer),
            patch(
                "controllers.api.health_controller.BaseConsumer.health_check",
                return_value=True,
            ),
        ):
            client = TestClient(app)
            response = client.get("/health/")

        assert response.status_code == 200
        data = response.json()
        assert data["qdrant"] is True
        assert data["gemini"] is True
        assert data["amqp"] is True

    def test_returns_503_when_qdrant_unhealthy(self) -> None:
        app = _make_app()
        mock_qdrant = MagicMock()
        mock_gemini = MagicMock()
        mock_consumer = MagicMock()
        mock_consumer.amqp_url = "amqp://localhost/"

        mock_qdrant.is_healthy = AsyncMock(return_value=False)
        mock_gemini.is_healthy.return_value = True

        with (
            patch_health_deps(app, mock_qdrant, mock_gemini, mock_consumer),
            patch(
                "controllers.api.health_controller.BaseConsumer.health_check",
                return_value=True,
            ),
        ):
            client = TestClient(app)
            response = client.get("/health/")

        assert response.status_code == 503
        data = response.json()
        assert data["qdrant"] is False
        assert data["gemini"] is True
        assert data["amqp"] is True

    def test_returns_503_when_gemini_unhealthy(self) -> None:
        app = _make_app()
        mock_qdrant = MagicMock()
        mock_gemini = MagicMock()
        mock_consumer = MagicMock()
        mock_consumer.amqp_url = "amqp://localhost/"

        mock_qdrant.is_healthy = AsyncMock(return_value=True)
        mock_gemini.is_healthy.return_value = False

        with (
            patch_health_deps(app, mock_qdrant, mock_gemini, mock_consumer),
            patch(
                "controllers.api.health_controller.BaseConsumer.health_check",
                return_value=True,
            ),
        ):
            client = TestClient(app)
            response = client.get("/health/")

        assert response.status_code == 503
        data = response.json()
        assert data["qdrant"] is True
        assert data["gemini"] is False
        assert data["amqp"] is True

    def test_returns_503_when_amqp_unhealthy(self) -> None:
        app = _make_app()
        mock_qdrant = MagicMock()
        mock_gemini = MagicMock()
        mock_consumer = MagicMock()
        mock_consumer.amqp_url = "amqp://localhost/"

        mock_qdrant.is_healthy = AsyncMock(return_value=True)
        mock_gemini.is_healthy.return_value = True

        with (
            patch_health_deps(app, mock_qdrant, mock_gemini, mock_consumer),
            patch(
                "controllers.api.health_controller.BaseConsumer.health_check",
                return_value=False,
            ),
        ):
            client = TestClient(app)
            response = client.get("/health/")

        assert response.status_code == 503
        data = response.json()
        assert data["qdrant"] is True
        assert data["gemini"] is True
        assert data["amqp"] is False

    def test_response_shape_matches_health_response_model(self) -> None:
        app = _make_app()
        mock_qdrant = MagicMock()
        mock_gemini = MagicMock()
        mock_consumer = MagicMock()
        mock_consumer.amqp_url = "amqp://localhost/"

        mock_qdrant.is_healthy = AsyncMock(return_value=True)
        mock_gemini.is_healthy.return_value = True

        with (
            patch_health_deps(app, mock_qdrant, mock_gemini, mock_consumer),
            patch(
                "controllers.api.health_controller.BaseConsumer.health_check",
                return_value=True,
            ),
        ):
            client = TestClient(app)
            response = client.get("/health/")

        data = response.json()
        parsed = HealthResponse.model_validate(data)
        assert parsed.qdrant is True
        assert parsed.gemini is True
        assert parsed.amqp is True


@contextmanager
def patch_health_deps(
    app: FastAPI,
    qdrant: QdrantClientService,
    gemini: GeminiClientService,
    consumer: BaseConsumer,
) -> Generator[None, None, None]:
    """Override health dependencies to use the given mocks."""

    def get_qdrant_override(request: Request) -> QdrantClientService:
        return qdrant

    def get_gemini_override(request: Request) -> GeminiClientService:
        return gemini

    def get_consumer_override(request: Request) -> BaseConsumer:
        return consumer

    app.dependency_overrides[health_controller.get_qdrant] = get_qdrant_override
    app.dependency_overrides[health_controller.get_gemini] = get_gemini_override
    app.dependency_overrides[health_controller.get_consumer] = get_consumer_override
    try:
        yield
    finally:
        app.dependency_overrides.clear()
