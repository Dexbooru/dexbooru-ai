from typing import Any

from fastapi.routing import APIRouter

from utils.config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/similarity/posts/images/config", tags=["ml"])


@router.get("/")
def handle_get_similarity_post_configuration():
    settings = get_settings()

    image_similarity_configuration: dict[str, Any] = {
        "allowed_image_domains": settings.allowed_image_domain_destinations,
        "all_image_domains_allowed_in_search": settings.allow_all_image_domain_destinations,
    }

    return image_similarity_configuration
