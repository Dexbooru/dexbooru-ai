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

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


# Singleton instance for app use (e.g. `from utils.config import settings`)
settings = ApplicationSettings()
