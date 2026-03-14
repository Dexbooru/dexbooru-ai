from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApplicationSettings(BaseSettings):
    server_name: str = Field(
        validation_alias="SERVER_NAME", description="The name of the server"
    )
    server_port: int = Field(
        validation_alias="SERVER_PORT", description="The port of the server"
    )
    amqp_url: str = Field(
        validation_alias="AMQP_URL", description="The URL of the AMQP server"
    )
    environment: str = Field(
        validation_alias="ENVIRONMENT",
        description="The environment of the server",
        default="development",
    )

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = ApplicationSettings()
