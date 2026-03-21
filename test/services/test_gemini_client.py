"""Unit tests for services.gemini_client.GeminiClientService."""

from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from models.application.posts import DexbooruPost
from services.gemini_client import GeminiClientService


def _settings_mock(**kwargs) -> MagicMock:
    defaults = {
        "gemini_api_key": "key",
        "gemini_embedding_model_name": "models/embed-001",
        "gemini_output_dimensions": 256,
        "gemini_request_timeout_seconds": 30.0,
        "gemini_vertex_location": None,
    }
    defaults.update(kwargs)
    return MagicMock(**defaults)


class TestGeminiClientServiceInit:
    """Tests for GeminiClientService.__init__."""

    def test_reads_settings_and_sets_attributes(self) -> None:
        with patch("services.gemini_client.get_settings") as mock_get_settings:
            mock_get_settings.return_value = _settings_mock(
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
            mock_get_settings.return_value = _settings_mock(
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
            mock_get_settings.return_value = _settings_mock(
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
            mock_get_settings.return_value = _settings_mock(
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
            mock_get_settings.return_value = _settings_mock(
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


def _make_gemini_service(output_dimensions: int = 256):
    """Create a GeminiClientService with mocked base and settings."""
    with patch("services.gemini_client.get_settings") as mock_get_settings:
        mock_get_settings.return_value = _settings_mock(
            gemini_output_dimensions=output_dimensions,
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
            return GeminiClientService()


class TestGeminiClientServiceNormalizeImageEmbedding:
    """Tests for GeminiClientService._noramlize_image_embedding."""

    def test_returns_embedding_unchanged_when_dims_not_equal_to_output(self) -> None:
        svc = _make_gemini_service(output_dimensions=256)
        embedding = [1.0] * 128
        result = svc._noramlize_image_embedding(embedding)
        assert result == embedding

    def test_returns_embedding_unchanged_when_dims_are_auto_normalized(self) -> None:
        svc = _make_gemini_service(output_dimensions=256)
        embedding = [1.0] * GeminiClientService.AUTO_NORMALIZED_EMBEDDING_DIMENSIONS
        result = svc._noramlize_image_embedding(embedding)
        assert result == embedding

    def test_normalizes_when_dims_equal_output_dimensions(self) -> None:
        svc = _make_gemini_service(output_dimensions=2)
        embedding = [3.0, 4.0]
        result = svc._noramlize_image_embedding(embedding)
        assert len(result) == 2
        assert abs(result[0] - 0.6) < 1e-5 and abs(result[1] - 0.8) < 1e-5

    def test_normalized_vector_has_unit_length(self) -> None:
        svc = _make_gemini_service(output_dimensions=3)
        embedding = [1.0, 0.0, 0.0]
        result = svc._noramlize_image_embedding(embedding)
        assert result == [1.0, 0.0, 0.0]


class TestGeminiClientServiceEmbedImages:
    """Tests for GeminiClientService.embed_images."""

    def test_raises_when_embedding_model_name_is_none(self) -> None:
        svc = _make_gemini_service()
        svc.embedding_model_name = None
        post = MagicMock(spec=DexbooruPost)
        post.id = "post-1"

        with pytest.raises(ValueError) as exc_info:
            svc.embed_images(post, [b"fake"])

        assert "Embedding model name is not set" in str(exc_info.value)

    def test_returns_empty_list_when_image_bytes_list_empty(self) -> None:
        svc = _make_gemini_service()
        post = MagicMock(spec=DexbooruPost)
        post.id = "post-1"

        result = svc.embed_images(post, [])

        assert result == []

    def test_embeds_inline_bytes_and_returns_normalized_embeddings(self) -> None:
        mock_embedding_response = MagicMock()
        mock_embedding_response.embeddings = [
            MagicMock(values=[0.1] * 256),
            MagicMock(values=[0.2] * 256),
        ]
        mock_models = MagicMock()
        mock_models.embed_content.return_value = mock_embedding_response

        with (
            patch("services.gemini_client.get_settings") as mock_get_settings,
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
            mock_get_settings.return_value = _settings_mock(
                gemini_embedding_model_name="models/embed-001",
                gemini_output_dimensions=256,
            )
            svc = GeminiClientService()
            post = MagicMock(spec=DexbooruPost)
            post.id = "post-1"
            image_bytes_list = [b"fake_png_1", b"fake_png_2"]
            result = svc.embed_images(post, image_bytes_list)

        assert len(result) == 2
        assert len(result[0]) == 256
        assert len(result[1]) == 256
        mock_models.embed_content.assert_called_once()
        call_kw = mock_models.embed_content.call_args[1]
        assert call_kw["model"] == "models/embed-001"
        assert len(call_kw["contents"]) == 2
        assert all(p.inline_data.mime_type == "image/png" for p in call_kw["contents"])
        assert call_kw["config"].output_dimensionality == 256
