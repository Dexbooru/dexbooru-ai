from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApplicationSettings(BaseSettings):
    # Server settings
    server_name: str = Field(validation_alias="SERVER_NAME")
    server_port: int = Field(validation_alias="SERVER_PORT", default=8001)
    environment: str = Field(default="development", validation_alias="ENVIRONMENT")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    # AMQP settings
    amqp_url: str = Field(validation_alias="AMQP_URL")
    primary_exchange_name: str = Field(
        default="ai_events", validation_alias="PRIMARY_EXCHANGE_NAME"
    )

    # Qdrant settings
    qdrant_url: str = Field(validation_alias="QDRANT_URL")
    qdrant_api_key: str = Field(validation_alias="QDRANT_API_KEY")

    # Gemini settings
    gemini_api_key: str = Field(validation_alias="GEMINI_API_KEY")
    gemini_embedding_model_name: str = Field(
        default="gemini-embedding-2-preview",
        validation_alias="GEMINI_EMBEDDING_MODEL_NAME",
    )
    gemini_output_dimensions: int = Field(
        default=1536, validation_alias="GEMINI_OUTPUT_DIMENSIONS"
    )

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


_instance: ApplicationSettings | None = None


def get_settings() -> ApplicationSettings:
    global _instance
    if _instance is None:
        _instance = ApplicationSettings()
    return _instance


def __getattr__(name: str) -> Any:
    if name == "settings":
        return get_settings()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
