import asyncio
import io

import aiohttp
from PIL import Image

from models.application.posts import DexbooruPost
from utils.config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


class ImagePreprocessor:
    TARGET_IMAGE_MIME_TYPE: str = "png"
    IMAGE_MIMETYPES: list[str] = ["image/jpeg", "image/jpg", "image/png", "image/webp", "webp", "png", "jpg", "jpeg"]

    def __init__(self, post: DexbooruPost):
        settings = get_settings()
        self.post = post
        self.cdn_base_url = settings.cdn_base_url.rstrip("/")
        self.image_resize_width = settings.image_resize_width
        self.image_resize_height = settings.image_resize_height

    def _is_mimetype_supported_image(self, mime_type: str) -> bool:
        return mime_type in ImagePreprocessor.IMAGE_MIMETYPES

    def _is_url_from_cdn(self, image_url: str) -> bool:
        return image_url.startswith(self.cdn_base_url)

    def _resize_to_dimensions(self, image_data: bytes) -> bytes:
        image = Image.open(io.BytesIO(image_data))
        image = image.resize((self.image_resize_width, self.image_resize_height))
        buffer = io.BytesIO()
        image.save(buffer, format=ImagePreprocessor.TARGET_IMAGE_MIME_TYPE.upper())
        return buffer.getvalue()

    async def _download_image(self, session: aiohttp.ClientSession, image_url: str) -> tuple[bytes, bool]:
        async with session.get(image_url) as response:
            if response.status != 200:
                logger.warning(
                    "image download failed: status=%s url=%s",
                    response.status,
                    image_url,
                )
                return b"", False

            response_mime_type = response.headers.get("Content-Type", "").lower().strip()
            if not self._is_mimetype_supported_image(response_mime_type):
                logger.warning(
                    "image download skipped: unsupported mimetype=%s url=%s",
                    response_mime_type or "(none)",
                    image_url,
                )
                return b"", False

            image_data = await response.read()
            logger.debug("image downloaded url=%s size=%s", image_url, len(image_data))
            return image_data, True

    async def transform(self) -> list[bytes]:
        logger.info(
            "transform started post_id=%s image_count=%s",
            self.post.id,
            len(self.post.image_urls),
        )

        urls_to_download = [url for url in self.post.image_urls if self._is_url_from_cdn(url)]
        skipped = len(self.post.image_urls) - len(urls_to_download)
        if skipped:
            logger.warning(
                "skipped %s image url(s) not from CDN post_id=%s",
                skipped,
                self.post.id,
            )

        if not urls_to_download:
            logger.info("transform finished post_id=%s output_count=0", self.post.id)
            return []

        async with aiohttp.ClientSession() as session:
            download_results = await asyncio.gather(*[self._download_image(session, url) for url in urls_to_download])

        result: list[bytes] = []
        failed_downloads = 0
        for image_data, success in download_results:
            if not success:
                failed_downloads += 1
                continue
            resized = self._resize_to_dimensions(image_data)
            result.append(resized)
            logger.debug("resized image size=%s", len(resized))

        if failed_downloads:
            logger.warning(
                "transform had %s failed download(s) post_id=%s",
                failed_downloads,
                self.post.id,
            )
        logger.info(
            "transform finished post_id=%s output_count=%s",
            self.post.id,
            len(result),
        )
        return result
