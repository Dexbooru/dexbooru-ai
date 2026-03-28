from pathlib import Path
from typing import Any

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _parse_cors_origins_str(raw: str) -> list[str]:
    if not raw or not raw.strip():
        return ["*"]
    s = raw.strip()
    if s == "*":
        return ["*"]
    return [x.strip() for x in s.split(",") if x.strip()]


class ApplicationSettings(BaseSettings):
    # Server settings
    server_name: str = Field(validation_alias="SERVER_NAME")
    server_port: int = Field(validation_alias="SERVER_PORT", default=8001)
    cors_origins_str: str = Field(default="*", validation_alias="CORS_ORIGINS")
    environment: str = Field(default="development", validation_alias="ENVIRONMENT")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    # AMQP settings
    amqp_url: str = Field(validation_alias="AMQP_URL")
    primary_exchange_name: str = Field(default="ai_events", validation_alias="PRIMARY_EXCHANGE_NAME")

    # Qdrant settings
    qdrant_url: str = Field(validation_alias="QDRANT_URL")
    qdrant_api_key: str = Field(validation_alias="QDRANT_API_KEY")

    # Gemini settings
    gemini_api_key: str = Field(validation_alias="GEMINI_API_KEY")
    gemini_embedding_model_name: str = Field(
        default="gemini-embedding-2-preview",
        validation_alias="GEMINI_EMBEDDING_MODEL_NAME",
    )
    gemini_output_dimensions: int = Field(default=1536, validation_alias="GEMINI_OUTPUT_DIMENSIONS")

    # image settings
    cdn_base_url: str = Field(validation_alias="CDN_BASE_URL")
    image_resize_width: int = Field(validation_alias="IMAGE_RESIZE_WIDTH", default=512)
    image_resize_height: int = Field(validation_alias="IMAGE_RESIZE_HEIGHT", default=512)

    # ML / spaCy (tag-rating predictor)
    danbooru_tag_rating_skops_path: Path = Field(validation_alias="DANBOORU_TAG_RATING_SKOPS_PATH")
    spacy_english_model: str = Field(default="en_core_web_md", validation_alias="SPACY_ENGLISH_MODEL")

    @computed_field
    @property
    def cors_origins(self) -> list[str]:
        return _parse_cors_origins_str(self.cors_origins_str)

    # environment configuration (system wide takes precedence over local)
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


_instance: ApplicationSettings | None = None


def get_settings() -> ApplicationSettings:
    global _instance
    if _instance is None:
        _instance = ApplicationSettings()
    return _instance


def reset_settings_cache() -> None:
    """Clear the memoized settings singleton (for tests / reload)."""
    global _instance
    _instance = None


def __getattr__(name: str) -> Any:
    if name == "settings":
        return get_settings()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
