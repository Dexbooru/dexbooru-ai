"""Unit tests for services.qdrant_client.QdrantClientService."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.qdrant_client import QdrantClientService


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

            with patch(
                "services.qdrant_client.AsyncQdrantClient",
                return_value=MagicMock(),
            ):
                svc = QdrantClientService()

        assert svc.output_dimensions == 768


class TestQdrantClientServiceGetClientBaseUrl:
    """Tests for QdrantClientService._get_client_base_url."""

    def test_returns_host_without_trailing_slash(self) -> None:
        with patch("services.qdrant_client.get_settings") as mock_get_settings:
            mock_get_settings.return_value = MagicMock(
                qdrant_url="http://localhost:6333",
                qdrant_api_key=None,
                environment="development",
                gemini_output_dimensions=1536,
            )
            with patch(
                "services.qdrant_client.AsyncQdrantClient",
                return_value=MagicMock(),
            ):
                svc = QdrantClientService()

        svc.client.http.client.host = "http://localhost:6333/"
        assert svc._get_client_base_url() == "http://localhost:6333"

        svc.client.http.client.host = "http://qdrant:6333"
        assert svc._get_client_base_url() == "http://qdrant:6333"


class TestQdrantClientServiceIsHealthy:
    """Tests for QdrantClientService.is_healthy."""

    def test_returns_true_when_healthz_200_and_expected_text(self) -> None:
        with patch("services.qdrant_client.get_settings") as mock_get_settings:
            mock_get_settings.return_value = MagicMock(
                qdrant_url="http://localhost:6333",
                qdrant_api_key=None,
                environment="development",
                gemini_output_dimensions=1536,
            )
            with patch(
                "services.qdrant_client.AsyncQdrantClient",
                return_value=MagicMock(),
            ):
                svc = QdrantClientService()
        svc.client.http.client.host = "http://localhost:6333"

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(
            return_value=QdrantClientService.HEALTH_CHECK_PASSED_TEXT
        )

        with patch("services.qdrant_client.aiohttp.ClientSession") as mock_session:
            cm = MagicMock()
            cm.__aenter__ = AsyncMock(return_value=mock_response)
            cm.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value.__aenter__.return_value.get = MagicMock(
                return_value=cm
            )

            result = asyncio.run(svc.is_healthy())

        assert result is True

    def test_returns_false_when_healthz_non_200(self) -> None:
        with patch("services.qdrant_client.get_settings") as mock_get_settings:
            mock_get_settings.return_value = MagicMock(
                qdrant_url="http://localhost:6333",
                qdrant_api_key=None,
                environment="development",
                gemini_output_dimensions=1536,
            )
            with patch(
                "services.qdrant_client.AsyncQdrantClient",
                return_value=MagicMock(),
            ):
                svc = QdrantClientService()
        svc.client.http.client.host = "http://localhost:6333"

        mock_response = AsyncMock()
        mock_response.status = 503
        mock_response.text = AsyncMock(return_value="unavailable")

        with patch("services.qdrant_client.aiohttp.ClientSession") as mock_session:
            cm = MagicMock()
            cm.__aenter__ = AsyncMock(return_value=mock_response)
            cm.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value.__aenter__.return_value.get = MagicMock(
                return_value=cm
            )

            result = asyncio.run(svc.is_healthy())

        assert result is False

    def test_returns_false_when_response_text_mismatch(self) -> None:
        with patch("services.qdrant_client.get_settings") as mock_get_settings:
            mock_get_settings.return_value = MagicMock(
                qdrant_url="http://localhost:6333",
                qdrant_api_key=None,
                environment="development",
                gemini_output_dimensions=1536,
            )
            with patch(
                "services.qdrant_client.AsyncQdrantClient",
                return_value=MagicMock(),
            ):
                svc = QdrantClientService()
        svc.client.http.client.host = "http://localhost:6333"

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="something else")

        with patch("services.qdrant_client.aiohttp.ClientSession") as mock_session:
            cm = MagicMock()
            cm.__aenter__ = AsyncMock(return_value=mock_response)
            cm.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value.__aenter__.return_value.get = MagicMock(
                return_value=cm
            )

            result = asyncio.run(svc.is_healthy())

        assert result is False

    def test_raises_on_network_error(self) -> None:
        with patch("services.qdrant_client.get_settings") as mock_get_settings:
            mock_get_settings.return_value = MagicMock(
                qdrant_url="http://localhost:6333",
                qdrant_api_key=None,
                environment="development",
                gemini_output_dimensions=1536,
            )
            with patch(
                "services.qdrant_client.AsyncQdrantClient",
                return_value=MagicMock(),
            ):
                svc = QdrantClientService()
        svc.client.http.client.host = "http://localhost:6333"

        with patch("services.qdrant_client.aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__.return_value.get = MagicMock(
                side_effect=OSError("connection refused")
            )

            with pytest.raises(OSError):
                asyncio.run(svc.is_healthy())
