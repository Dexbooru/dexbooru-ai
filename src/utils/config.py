from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApplicationSettings(BaseSettings):
    # Server settings
    server_name: str = Field(
        validation_alias="SERVER_NAME", description="The name of the server"
    )
    server_port: int = Field(
        validation_alias="SERVER_PORT", description="The port of the server"
    )
    environment: str = Field(
        validation_alias="ENVIRONMENT",
        description="The environment of the server",
        default="development",
    )
    log_level: str = Field(
        validation_alias="LOG_LEVEL",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        default="INFO",
    )

    # AMQP settings
    amqp_url: str = Field(
        validation_alias="AMQP_URL", description="The URL of the AMQP server"
    )
    primary_exchange_name: str = Field(
        validation_alias="PRIMARY_EXCHANGE_NAME",
        description="The AMQP exchange all consumers read from",
        default="ai_events",
    )

    # Qdrant settings
    qdrant_url: str = Field(
        validation_alias="QDRANT_URL", description="The URL of the Qdrant server"
    )
    qdrant_api_key: str = Field(
        validation_alias="QDRANT_API_KEY",
        description="The API key of the Qdrant server",
    )

    # Gemini settings
    gemini_api_key: str = Field(
        validation_alias="GEMINI_API_KEY",
        description="The API key of the Gemini server",
    )
    gemini_embedding_model_name: str = Field(
        validation_alias="GEMINI_EMBEDDING_MODEL_NAME",
        description="The name of the Gemini embedding model",
        default="gemini-embedding-2-preview",
    )
    gemini_output_dimensions: int = Field(
        validation_alias="GEMINI_OUTPUT_DIMENSIONS",
        description="The output dimensions of the Gemini model",
        default=1536,
    )

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


# Singleton instance - create once, reuse everywhere
_instance: ApplicationSettings | None = None


def get_settings() -> ApplicationSettings:
    """Return the application settings singleton. Use anywhere via:
    from utils.config import get_settings
    settings = get_settings()
    """
    global _instance
    if _instance is None:
        _instance = ApplicationSettings()
    return _instance


# Convenience alias so `from utils.config import settings` works.
def __getattr__(name: str) -> Any:
    if name == "settings":
        return get_settings()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
