import asyncio
import io
from datetime import UTC, datetime
from uuid import uuid4

from PIL import Image

from models.application.posts import DexbooruPost
from utils.image_preprecessor import (
    ImagePreprocessor,
)


def _make_png_bytes(width: int = 10, height: int = 10) -> bytes:
    """Create minimal PNG image bytes."""
    img = Image.new("RGB", (width, height), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_post(
    *,
    image_urls: list[str] | None = None,
    post_id: str | None = None,
) -> DexbooruPost:
    now = datetime.now(UTC)
    return DexbooruPost(
        id=post_id or uuid4(),
        description="test post",
        image_urls=image_urls or [],
        created_at=now,
        updated_at=now,
        author_id=uuid4(),
    )


def _preprocessor_with_settings(
    post: DexbooruPost,
    cdn_base_url: str = "https://cdn.example.com",
    image_resize_width: int = 64,
    image_resize_height: int = 64,
):
    from unittest.mock import MagicMock, patch

    with patch("utils.image_preprecessor.get_settings") as mock_get_settings:
        mock_get_settings.return_value = MagicMock(
            cdn_base_url=cdn_base_url,
            image_resize_width=image_resize_width,
            image_resize_height=image_resize_height,
        )
        return ImagePreprocessor(post)


class TestImagePreprocessorIsMimetypeSupportedImage:
    """Tests for _is_mimetype_supported_image."""

    def test_returns_true_for_supported_mimetypes(self) -> None:
        post = _make_post()
        preprocessor = _preprocessor_with_settings(post)
        for mime in ImagePreprocessor.IMAGE_MIMETYPES:
            assert preprocessor._is_mimetype_supported_image(mime) is True

    def test_returns_false_for_unsupported_mimetype(self) -> None:
        post = _make_post()
        preprocessor = _preprocessor_with_settings(post)
        assert preprocessor._is_mimetype_supported_image("image/bmp") is False
        assert preprocessor._is_mimetype_supported_image("text/plain") is False
        assert preprocessor._is_mimetype_supported_image("") is False


class TestImagePreprocessorIsUrlFromCdn:
    """Tests for _is_url_from_cdn."""

    def test_returns_true_when_url_starts_with_cdn_base(self) -> None:
        post = _make_post()
        preprocessor = _preprocessor_with_settings(post, cdn_base_url="https://cdn.example.com")
        assert preprocessor._is_url_from_cdn("https://cdn.example.com/img/1.png") is True
        assert preprocessor._is_url_from_cdn("https://cdn.example.com/path/to/file.jpg") is True

    def test_returns_false_when_url_does_not_start_with_cdn_base(self) -> None:
        post = _make_post()
        preprocessor = _preprocessor_with_settings(post, cdn_base_url="https://cdn.example.com")
        assert preprocessor._is_url_from_cdn("https://other.example.com/img/1.png") is False
        assert preprocessor._is_url_from_cdn("https://cdn.evil.com/img/1.png") is False
        assert preprocessor._is_url_from_cdn("") is False


class TestImagePreprocessorResizeToDimensions:
    """Tests for resize_image_bytes (wraps _resize_to_dimensions)."""

    def test_returns_png_bytes_resized_to_dimensions(self) -> None:
        post = _make_post()
        preprocessor = _preprocessor_with_settings(post, image_resize_width=8, image_resize_height=8)
        image_data = _make_png_bytes(100, 100)

        result = preprocessor.resize_image_bytes(image_data)

        assert isinstance(result, bytes)
        assert len(result) > 0
        loaded = Image.open(io.BytesIO(result))
        assert loaded.size == (8, 8)
        assert loaded.format == "PNG"

    def test_output_is_valid_png(self) -> None:
        post = _make_post()
        preprocessor = _preprocessor_with_settings(post)
        image_data = _make_png_bytes(5, 5)

        result = preprocessor.resize_image_bytes(image_data)

        assert isinstance(result, bytes)
        img = Image.open(io.BytesIO(result))
        assert img.size == (64, 64)
        assert img.format == "PNG"


def _make_mock_session(get_return_value):
    """Build a mock aiohttp.ClientSession whose .get() returns the given context manager."""
    from unittest.mock import MagicMock

    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=get_return_value)
    return mock_session


class TestImagePreprocessorDownloadImage:
    """Tests for _download_image. Session is reused by transform(); tests pass a mock session."""

    def test_returns_data_and_true_on_200_and_supported_mimetype(self) -> None:
        from unittest.mock import AsyncMock, MagicMock

        post = _make_post()
        preprocessor = _preprocessor_with_settings(post)
        image_data = _make_png_bytes(2, 2)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "image/png"}
        mock_response.read = AsyncMock(return_value=image_data)

        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=mock_response)
        cm.__aexit__ = AsyncMock(return_value=None)

        session = _make_mock_session(cm)
        result = asyncio.run(preprocessor._download_image(session, "https://cdn.example.com/img.png"))

        data, success = result
        assert success is True
        assert data == image_data
        session.get.assert_called_once_with("https://cdn.example.com/img.png")

    def test_returns_empty_and_false_on_non_200(self) -> None:
        from unittest.mock import AsyncMock, MagicMock

        post = _make_post()
        preprocessor = _preprocessor_with_settings(post)

        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.headers = {}

        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=mock_response)
        cm.__aexit__ = AsyncMock(return_value=None)

        session = _make_mock_session(cm)
        data, success = asyncio.run(preprocessor._download_image(session, "https://cdn.example.com/missing.png"))

        assert success is False
        assert data == b""

    def test_returns_empty_and_false_on_unsupported_mimetype(self) -> None:
        from unittest.mock import AsyncMock, MagicMock

        post = _make_post()
        preprocessor = _preprocessor_with_settings(post)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "image/bmp"}
        mock_response.read = AsyncMock(return_value=b"fake")

        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=mock_response)
        cm.__aexit__ = AsyncMock(return_value=None)

        session = _make_mock_session(cm)
        data, success = asyncio.run(preprocessor._download_image(session, "https://cdn.example.com/img.bmp"))

        assert success is False
        assert data == b""


class TestImagePreprocessorTransform:
    """Tests for transform. One ClientSession per post, downloads run concurrently."""

    def test_creates_one_session_and_downloads_concurrently(self) -> None:
        from unittest.mock import AsyncMock, MagicMock, patch

        post = _make_post(
            image_urls=[
                "https://cdn.example.com/a.png",
                "https://cdn.example.com/b.png",
            ],
        )
        preprocessor = _preprocessor_with_settings(post, cdn_base_url="https://cdn.example.com")
        image_data = _make_png_bytes(3, 3)

        def make_fake_cm():
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.headers = {"Content-Type": "image/png"}
            mock_resp.read = AsyncMock(return_value=image_data)
            cm = MagicMock()
            cm.__aenter__ = AsyncMock(return_value=mock_resp)
            cm.__aexit__ = AsyncMock(return_value=None)
            return cm

        with patch("utils.image_preprecessor.aiohttp.ClientSession") as mock_cls:
            mock_session = MagicMock()
            mock_session.get = MagicMock(side_effect=lambda *a, **k: make_fake_cm())
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=None)

            result = asyncio.run(preprocessor.transform())

        assert len(result) == 2
        assert mock_cls.return_value.__aenter__.await_count == 1
        assert mock_session.get.call_count == 2

    def test_returns_empty_list_when_post_has_no_image_urls(self) -> None:
        post = _make_post(image_urls=[])
        preprocessor = _preprocessor_with_settings(post)

        result = asyncio.run(preprocessor.transform())

        assert result == []

    def test_skips_urls_not_from_cdn_and_processes_cdn_urls(self) -> None:
        from unittest.mock import AsyncMock, MagicMock, patch

        post = _make_post(
            image_urls=[
                "https://cdn.example.com/valid.png",
                "https://other.com/skip.png",
            ],
        )
        preprocessor = _preprocessor_with_settings(post, cdn_base_url="https://cdn.example.com")
        image_data = _make_png_bytes(4, 4)

        def make_fake_cm():
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.headers = {"Content-Type": "image/png"}
            mock_resp.read = AsyncMock(return_value=image_data)
            cm = MagicMock()
            cm.__aenter__ = AsyncMock(return_value=mock_resp)
            cm.__aexit__ = AsyncMock(return_value=None)
            return cm

        with patch("utils.image_preprecessor.aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__.return_value.get = lambda *a, **k: make_fake_cm()

            result = asyncio.run(preprocessor.transform())

        assert len(result) == 1
        assert isinstance(result[0], bytes)
        img = Image.open(io.BytesIO(result[0]))
        assert img.size == (64, 64)
        assert img.format == "PNG"

    def test_returns_list_of_bytes_for_successful_downloads(self) -> None:
        from unittest.mock import AsyncMock, MagicMock, patch

        post = _make_post(
            image_urls=[
                "https://cdn.example.com/a.png",
                "https://cdn.example.com/b.png",
            ],
        )
        preprocessor = _preprocessor_with_settings(post, cdn_base_url="https://cdn.example.com")
        image_data = _make_png_bytes(3, 3)

        def make_fake_cm():
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.headers = {"Content-Type": "image/png"}
            mock_resp.read = AsyncMock(return_value=image_data)
            cm = MagicMock()
            cm.__aenter__ = AsyncMock(return_value=mock_resp)
            cm.__aexit__ = AsyncMock(return_value=None)
            return cm

        with patch("utils.image_preprecessor.aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__.return_value.get = lambda *a, **k: make_fake_cm()

            result = asyncio.run(preprocessor.transform())

        assert len(result) == 2
        for img_bytes in result:
            assert isinstance(img_bytes, bytes)
            img = Image.open(io.BytesIO(img_bytes))
            assert img.size == (64, 64)
            assert img.format == "PNG"

    def test_ignores_failed_downloads_and_returns_only_successful_bytes(self) -> None:
        from unittest.mock import AsyncMock, MagicMock, patch

        post = _make_post(
            image_urls=[
                "https://cdn.example.com/ok.png",
                "https://cdn.example.com/fail.png",
            ],
        )
        preprocessor = _preprocessor_with_settings(post, cdn_base_url="https://cdn.example.com")
        image_data = _make_png_bytes(2, 2)

        def make_fake_get(url=None, *args, **kwargs):
            mock_resp = AsyncMock()
            if url and "fail" in url:
                mock_resp.status = 404
                mock_resp.headers = {}
            else:
                mock_resp.status = 200
                mock_resp.headers = {"Content-Type": "image/png"}
                mock_resp.read = AsyncMock(return_value=image_data)
            cm = MagicMock()
            cm.__aenter__ = AsyncMock(return_value=mock_resp)
            cm.__aexit__ = AsyncMock(return_value=None)
            return cm

        with patch("utils.image_preprecessor.aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__.return_value.get = make_fake_get

            result = asyncio.run(preprocessor.transform())

        assert len(result) == 1
        assert isinstance(result[0], bytes)
