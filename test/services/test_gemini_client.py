"""Unit tests for services.gemini_client.GeminiClientService."""

from unittest.mock import MagicMock, PropertyMock, patch

from services.gemini_client import GeminiClientService


class TestGeminiClientServiceInit:
    """Tests for GeminiClientService.__init__."""

    def test_reads_settings_and_sets_attributes(self) -> None:
        with patch("services.gemini_client.get_settings") as mock_get_settings:
            mock_get_settings.return_value = MagicMock(
                gemini_api_key="test-key",
                gemini_embedding_model_name="models/embed-001",
                gemini_output_dimensions=256,
            )
            with (
                patch(
                    "services.gemini_client.genai.Client.__init__",
                    return_value=None,
                ),
                patch(
                    "services.gemini_client.genai.Client.models",
                    new_callable=PropertyMock,
                    return_value=MagicMock(),
                ),
            ):
                svc = GeminiClientService()

        assert svc.embedding_model_name == "models/embed-001"
        assert svc.output_dimensions == 256


class TestGeminiClientServiceIsHealthy:
    """Tests for GeminiClientService.is_healthy."""

    def test_returns_true_when_models_list_returns_valid_response(self) -> None:
        mock_models = MagicMock()
        mock_response = MagicMock()
        mock_response.sdk_http_response.headers = {"content-type": "application/json"}
        mock_response.config = MagicMock()
        mock_models.list.return_value = mock_response

        with patch("services.gemini_client.get_settings") as mock_get_settings:
            mock_get_settings.return_value = MagicMock(
                gemini_api_key="key",
                gemini_embedding_model_name="embed",
                gemini_output_dimensions=1536,
            )
            with (
                patch(
                    "services.gemini_client.genai.Client.__init__",
                    return_value=None,
                ),
                patch(
                    "services.gemini_client.genai.Client.models",
                    new_callable=PropertyMock,
                    return_value=mock_models,
                ),
            ):
                svc = GeminiClientService()
                result = svc.is_healthy()

        assert result is True

    def test_returns_false_when_headers_is_none(self) -> None:
        mock_models = MagicMock()
        mock_response = MagicMock()
        mock_response.sdk_http_response.headers = None
        mock_response.config = MagicMock()
        mock_models.list.return_value = mock_response

        with patch("services.gemini_client.get_settings") as mock_get_settings:
            mock_get_settings.return_value = MagicMock(
                gemini_api_key="key",
                gemini_embedding_model_name="embed",
                gemini_output_dimensions=1536,
            )
            with (
                patch(
                    "services.gemini_client.genai.Client.__init__",
                    return_value=None,
                ),
                patch(
                    "services.gemini_client.genai.Client.models",
                    new_callable=PropertyMock,
                    return_value=mock_models,
                ),
            ):
                svc = GeminiClientService()
                result = svc.is_healthy()

        assert result is False

    def test_returns_false_when_config_is_none(self) -> None:
        mock_models = MagicMock()
        mock_response = MagicMock()
        mock_response.sdk_http_response.headers = {}
        mock_response.config = None
        mock_models.list.return_value = mock_response

        with patch("services.gemini_client.get_settings") as mock_get_settings:
            mock_get_settings.return_value = MagicMock(
                gemini_api_key="key",
                gemini_embedding_model_name="embed",
                gemini_output_dimensions=1536,
            )
            with (
                patch(
                    "services.gemini_client.genai.Client.__init__",
                    return_value=None,
                ),
                patch(
                    "services.gemini_client.genai.Client.models",
                    new_callable=PropertyMock,
                    return_value=mock_models,
                ),
            ):
                svc = GeminiClientService()
                result = svc.is_healthy()

        assert result is False


class TestGeminiClientServiceBuildEmbeddingModelConfig:
    """Tests for GeminiClientService._build_embedding_model_config."""

    def test_returns_config_with_output_dimensionality(self) -> None:
        with patch("services.gemini_client.get_settings") as mock_get_settings:
            mock_get_settings.return_value = MagicMock(
                gemini_api_key="key",
                gemini_embedding_model_name="embed",
                gemini_output_dimensions=512,
            )
            with (
                patch(
                    "services.gemini_client.genai.Client.__init__",
                    return_value=None,
                ),
                patch(
                    "services.gemini_client.genai.Client.models",
                    new_callable=PropertyMock,
                    return_value=MagicMock(),
                ),
            ):
                svc = GeminiClientService()
                config = svc._build_embedding_model_config()

        assert config.output_dimensionality == 512
