"""Unit tests for services.gemini_client.GeminiClientService."""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from unittest.mock import MagicMock, PropertyMock, patch
from uuid import uuid4

import pytest
from google.genai import errors

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


class TestGeminiClientServiceNormalizeEmbedding:
    """Tests for GeminiClientService._normalize_embedding."""

    def test_returns_embedding_unchanged_when_dims_not_equal_to_output(self) -> None:
        svc = _make_gemini_service(output_dimensions=256)
        embedding = [1.0] * 128
        result = svc._normalize_embedding(embedding)
        assert result == embedding

    def test_returns_embedding_unchanged_when_dims_are_auto_normalized(self) -> None:
        svc = _make_gemini_service(output_dimensions=256)
        embedding = [1.0] * GeminiClientService.AUTO_NORMALIZED_EMBEDDING_DIMENSIONS
        result = svc._normalize_embedding(embedding)
        assert result == embedding

    def test_normalizes_when_dims_equal_output_dimensions(self) -> None:
        svc = _make_gemini_service(output_dimensions=2)
        embedding = [3.0, 4.0]
        result = svc._normalize_embedding(embedding)
        assert len(result) == 2
        assert abs(result[0] - 0.6) < 1e-5 and abs(result[1] - 0.8) < 1e-5

    def test_normalized_vector_has_unit_length(self) -> None:
        svc = _make_gemini_service(output_dimensions=3)
        embedding = [1.0, 0.0, 0.0]
        result = svc._normalize_embedding(embedding)
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


def _make_post(
    description: str = "a cat",
    tags: list[str] | None = None,
    artists: list[str] | None = None,
) -> DexbooruPost:
    now = datetime.now(UTC)
    return DexbooruPost(
        id=uuid4(),
        description=description,
        image_urls=[],
        tags=tags or [],
        artists=artists or [],
        created_at=now,
        updated_at=now,
        author_id=uuid4(),
    )


@contextmanager
def _gemini_service_with_models(mock_models: MagicMock, output_dimensions: int = 4) -> Iterator[GeminiClientService]:
    with (
        patch("services.gemini_client.get_settings") as mock_get_settings,
        patch("services.gemini_client.genai.Client.__init__", return_value=None),
        patch("services.gemini_client.genai.Client.models", new_callable=PropertyMock, return_value=mock_models),
    ):
        mock_get_settings.return_value = _settings_mock(
            gemini_embedding_model_name="gemini-embedding-2",
            gemini_output_dimensions=output_dimensions,
        )
        yield GeminiClientService()


def _embedding_response(*vectors: list[float]) -> MagicMock:
    return MagicMock(embeddings=[MagicMock(values=vector) for vector in vectors])


class TestGeminiClientServiceBuildPostTextDocument:
    """Tests for GeminiClientService.build_post_text_document."""

    def test_formats_title_tags_and_artists(self) -> None:
        post = _make_post(description="Sunset at the beach", tags=["long_hair", "smile"], artists=["artist one"])

        document = GeminiClientService.build_post_text_document(post)

        assert document == "title: Sunset at the beach | text: tags: long hair, smile; artists: artist one"

    def test_cleans_whitespace_and_drops_blank_and_duplicate_terms(self) -> None:
        post = _make_post(
            description="  many   spaces\n here ",
            tags=["cat", " ", "cat", "cat_ears", "cat ears"],
            artists=["", "bob", "bob "],
        )

        document = GeminiClientService.build_post_text_document(post)

        assert document == "title: many spaces here | text: tags: cat, cat ears; artists: bob"

    def test_omits_empty_sections(self) -> None:
        assert GeminiClientService.build_post_text_document(_make_post(description="only title")) == "title: only title | text: none"
        assert GeminiClientService.build_post_text_document(_make_post(description="t", tags=["a"])) == "title: t | text: tags: a"
        assert GeminiClientService.build_post_text_document(_make_post(description="t", artists=["b"])) == "title: t | text: artists: b"

    def test_uses_none_title_when_description_blank(self) -> None:
        post = _make_post(description="   ", tags=["cat"])

        assert GeminiClientService.build_post_text_document(post) == "title: none | text: tags: cat"

    def test_returns_none_when_all_fields_blank(self) -> None:
        post = _make_post(description=" \t ", tags=[" ", "_"], artists=[""])

        assert GeminiClientService.build_post_text_document(post) is None


class TestGeminiClientServiceEmbedPostText:
    """Tests for GeminiClientService.embed_post_text."""

    def test_embeds_document_and_returns_single_vector(self) -> None:
        mock_models = MagicMock()
        mock_models.embed_content.return_value = _embedding_response([0.5, 0.5, 0.5, 0.5])

        with _gemini_service_with_models(mock_models) as svc:
            result = svc.embed_post_text(_make_post(description="a cat", tags=["cute"]))

        assert result == pytest.approx([0.5, 0.5, 0.5, 0.5])
        call_kw = mock_models.embed_content.call_args[1]
        assert call_kw["model"] == "gemini-embedding-2"
        assert call_kw["contents"] == ["title: a cat | text: tags: cute"]
        assert call_kw["config"].output_dimensionality == 4

    def test_returns_none_without_calling_api_when_no_text(self) -> None:
        mock_models = MagicMock()

        with _gemini_service_with_models(mock_models) as svc:
            result = svc.embed_post_text(_make_post(description=" "))

        assert result is None
        mock_models.embed_content.assert_not_called()

    def test_raises_when_embedding_model_name_is_none(self) -> None:
        with _gemini_service_with_models(MagicMock()) as svc:
            svc.embedding_model_name = None
            with pytest.raises(ValueError, match="Embedding model name is not set"):
                svc.embed_post_text(_make_post())

    @pytest.mark.parametrize("response", [MagicMock(embeddings=[]), MagicMock(embeddings=None), _embedding_response([])])
    def test_raises_when_response_has_no_embeddings(self, response: MagicMock) -> None:
        mock_models = MagicMock()
        mock_models.embed_content.return_value = response
        post = _make_post()

        with _gemini_service_with_models(mock_models) as svc:
            with pytest.raises(ValueError, match=f"no text embeddings for post {post.id}"):
                svc.embed_post_text(post)

    def test_raises_when_dimensions_mismatch(self) -> None:
        mock_models = MagicMock()
        mock_models.embed_content.return_value = _embedding_response([0.1, 0.2])

        with _gemini_service_with_models(mock_models, output_dimensions=4) as svc:
            with pytest.raises(ValueError, match=r"dimensions \[2\].*expected 4"):
                svc.embed_post_text(_make_post())

    def test_wraps_api_errors_with_post_context(self) -> None:
        mock_models = MagicMock()
        mock_models.embed_content.side_effect = errors.APIError(429, {"error": {"message": "quota", "status": "RESOURCE_EXHAUSTED"}})
        post = _make_post()

        with _gemini_service_with_models(mock_models) as svc:
            with pytest.raises(RuntimeError, match=f"post {post.id}.*status code 429") as exc_info:
                svc.embed_post_text(post)

        assert isinstance(exc_info.value.__cause__, errors.APIError)


class TestGeminiClientServiceEmbedImagesEdgeCases:
    """Edge cases for GeminiClientService.embed_images."""

    def test_drops_empty_bytes_before_embedding(self) -> None:
        mock_models = MagicMock()
        mock_models.embed_content.return_value = _embedding_response([0.5, 0.5, 0.5, 0.5])

        with _gemini_service_with_models(mock_models) as svc:
            result = svc.embed_images(_make_post(), [b"", b"png", b""])

        assert len(result) == 1
        assert len(mock_models.embed_content.call_args[1]["contents"]) == 1

    def test_returns_empty_list_when_all_bytes_empty(self) -> None:
        mock_models = MagicMock()

        with _gemini_service_with_models(mock_models) as svc:
            result = svc.embed_images(_make_post(), [b"", b""])

        assert result == []
        mock_models.embed_content.assert_not_called()
