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


def _parse_comma_separated_str(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


class ApplicationSettings(BaseSettings):
    # Server settings
    server_name: str = Field(validation_alias="SERVER_NAME")
    server_port: int = Field(validation_alias="SERVER_PORT", default=8001)
    cors_origins_str: str = Field(default="*", validation_alias="CORS_ORIGINS")
    allowed_image_domain_destinations_str: str = Field(default="*", validation_alias="ALLOWED_IMAGE_DOMAIN_DESTINATIONS")
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
        default="gemini-embedding-2",
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

    # Database settings
    database_url: str = Field(validation_alias="DATABASE_URL")
    database_pool_size: int = Field(default=10, gt=0, validation_alias="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=5, ge=0, validation_alias="DATABASE_MAX_OVERFLOW")
    database_pool_timeout_seconds: int = Field(default=10, gt=0, validation_alias="DATABASE_POOL_TIMEOUT_SECONDS")
    database_pool_recycle_seconds: int = Field(default=1800, gt=0, validation_alias="DATABASE_POOL_RECYCLE_SECONDS")

    # Redis settings
    redis_url: str = Field(validation_alias="REDIS_URL")
    redis_max_connections: int = Field(default=20, gt=0, validation_alias="REDIS_MAX_CONNECTIONS")

    # Replication settings (upstream URL is only used by the replica setup script)
    upstream_database_url: str | None = Field(default=None, validation_alias="UPSTREAM_DATABASE_URL")
    replication_publication: str = Field(default="dexbooru_ai_content", validation_alias="REPLICATION_PUBLICATION")
    replication_subscription: str = Field(default="dexbooru_ai_content_sub", validation_alias="REPLICATION_SUBSCRIPTION")
    replication_slot: str = Field(default="dexbooru_ai_content_slot", validation_alias="REPLICATION_SLOT")

    # Auth settings (JWT_SECRET must match dexbooru-web's JWT_PRIVATE_KEY)
    jwt_secret: str = Field(validation_alias="JWT_SECRET")
    jwt_issuer: str | None = Field(default=None, validation_alias="JWT_ISSUER")
    jwt_audience: str | None = Field(default=None, validation_alias="JWT_AUDIENCE")
    app_domain: str = Field(validation_alias="APP_DOMAIN")
    trusted_proxy_ips_str: str = Field(default="", validation_alias="TRUSTED_PROXY_IPS")

    # Chat model settings
    chat_model_base_url: str = Field(validation_alias="CHAT_MODEL_BASE_URL")
    chat_model_api_key: str = Field(validation_alias="CHAT_MODEL_API_KEY")
    chat_model_name: str = Field(validation_alias="CHAT_MODEL_NAME")
    chat_model_timeout_seconds: int = Field(default=60, gt=0, validation_alias="CHAT_MODEL_TIMEOUT_SECONDS")
    chat_request_limit: int = Field(default=8, gt=0, validation_alias="CHAT_REQUEST_LIMIT")
    chat_history_message_limit: int = Field(default=20, gt=0, validation_alias="CHAT_HISTORY_MESSAGE_LIMIT")
    chat_max_message_chars: int = Field(default=2000, gt=0, validation_alias="CHAT_MAX_MESSAGE_CHARS")

    # Chat rate limits and quotas
    chat_user_messages_per_minute: int = Field(default=6, gt=0, validation_alias="CHAT_USER_MESSAGES_PER_MINUTE")
    chat_ip_messages_per_minute: int = Field(default=20, gt=0, validation_alias="CHAT_IP_MESSAGES_PER_MINUTE")
    chat_user_reads_per_minute: int = Field(default=60, gt=0, validation_alias="CHAT_USER_READS_PER_MINUTE")
    auth_ip_requests_per_minute: int = Field(default=10, gt=0, validation_alias="AUTH_IP_REQUESTS_PER_MINUTE")
    chat_daily_thread_limit: int = Field(default=10, gt=0, validation_alias="CHAT_DAILY_THREAD_LIMIT")
    chat_daily_message_limit: int = Field(default=100, gt=0, validation_alias="CHAT_DAILY_MESSAGE_LIMIT")
    chat_daily_token_limit: int = Field(default=200000, gt=0, validation_alias="CHAT_DAILY_TOKEN_LIMIT")
    chat_thread_message_limit: int = Field(default=40, gt=0, validation_alias="CHAT_THREAD_MESSAGE_LIMIT")

    @computed_field
    @property
    def cors_origins(self) -> list[str]:
        return _parse_cors_origins_str(self.cors_origins_str)

    @computed_field
    @property
    def allowed_image_domain_destinations(self) -> list[str]:
        return _parse_cors_origins_str(self.allowed_image_domain_destinations_str)

    @computed_field
    @property
    def trusted_proxy_ips(self) -> list[str]:
        return _parse_comma_separated_str(self.trusted_proxy_ips_str)

    @property
    def allow_all_image_domain_destinations(self) -> bool:
        return len(self.allowed_image_domain_destinations) == 1 and self.allowed_image_domain_destinations[0] == "*"

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
