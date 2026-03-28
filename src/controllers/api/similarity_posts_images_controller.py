import aiohttp
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from core.dependencies import get_gemini, get_qdrant
from models.api_responses.post_image_similarity import (
    PostImageSimilarityResult,
    PostImageSimilaritySearchForm,
    PostImageSimilaritySearchResponse,
)
from models.application.posts import DexbooruPost
from services.gemini_client import GeminiClientService
from services.qdrant_client import QdrantClientService
from utils.image_preprecessor import ImagePreprocessor
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/similarity/posts/images", tags=["ml"])

MAX_UPLOAD_IMAGE_BYTES = 3 * 1024 * 1024


def _normalize_content_type(content_type: str | None) -> str:
    if not content_type:
        return ""
    return content_type.lower().split(";")[0].strip()


async def _load_image_from_url(image_url: str) -> bytes:
    async with aiohttp.ClientSession() as session:
        async with session.get(image_url) as response:
            if response.status != 200:
                raise HTTPException(status_code=400, detail=f"Failed to download image from URL: status {response.status}")

            mime_type = _normalize_content_type(response.headers.get("Content-Type"))
            if mime_type not in ImagePreprocessor.IMAGE_MIMETYPES:
                raise HTTPException(status_code=400, detail="Unsupported image MIME type for provided URL")

            image_data = await response.read()
            if not image_data:
                raise HTTPException(status_code=400, detail="Downloaded image is empty")
            return image_data


async def _load_image_from_upload(image_file: UploadFile) -> bytes:
    mime_type = _normalize_content_type(image_file.content_type)
    if mime_type not in ImagePreprocessor.IMAGE_MIMETYPES:
        raise HTTPException(status_code=400, detail="Unsupported uploaded image MIME type")

    image_data = await image_file.read()
    if len(image_data) > MAX_UPLOAD_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="Uploaded file exceeds 3MB limit")
    if not image_data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    return image_data


@router.post("/", response_model=PostImageSimilaritySearchResponse)
async def search_similar_post_images(
    image_url: str | None = Form(default=None),
    top_closest_match_count: int = Form(default=5),
    description: str | None = Form(default=None),
    image_file: UploadFile | None = File(default=None),
    qdrant: QdrantClientService = Depends(get_qdrant),
    gemini: GeminiClientService = Depends(get_gemini),
) -> PostImageSimilaritySearchResponse:
    desc_stripped = description.strip() if description else None
    form = PostImageSimilaritySearchForm(
        image_url=image_url.strip() if image_url else None,
        top_closest_match_count=top_closest_match_count,
        description=desc_stripped if desc_stripped else None,
    )

    has_image_url = bool(form.image_url)
    has_image_file = image_file is not None
    if has_image_url == has_image_file:
        logger.warning("similarity posts/images rejected | invalid image input combination")
        raise HTTPException(status_code=400, detail="Provide exactly one of image_url or image_file")

    source_type = "image_url" if has_image_url else "image_file"
    logger.info(
        "similarity posts/images request | source=%s | top_k=%d",
        source_type,
        form.top_closest_match_count,
    )

    try:
        input_image_bytes = await (
            _load_image_from_url(form.image_url) if form.image_url else _load_image_from_upload(image_file)  # type: ignore[arg-type]
        )
        synthetic_post = DexbooruPost.synthetic_similarity_search_post(description=form.description)
        preprocessor = ImagePreprocessor(synthetic_post)
        transformed_image = preprocessor.resize_image_bytes(input_image_bytes)

        embeddings = gemini.embed_images(synthetic_post, [transformed_image])
        if not embeddings:
            raise HTTPException(status_code=500, detail="Failed to generate image embedding")

        similar_results = await qdrant.search_post_image_similarity(
            query_vector=embeddings[0],
            limit=form.top_closest_match_count,
        )

        response_results = [
            PostImageSimilarityResult(
                post_id=result.post_id,
                image_url=result.image_url,
                similarity_score=round(result.score * 100, 2),
            )
            for result in similar_results
        ]
        logger.info(
            "similarity posts/images success | source=%s | top_k=%d | results=%d",
            source_type,
            form.top_closest_match_count,
            len(response_results),
        )
        return PostImageSimilaritySearchResponse(results=response_results)
    except HTTPException as e:
        logger.error(
            "similarity posts/images rejected | source=%s | status=%d | detail=%s",
            source_type,
            e.status_code,
            e.detail,
        )
        raise
    except Exception as e:
        logger.error("similarity posts/images failed | source=%s", source_type, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from e
