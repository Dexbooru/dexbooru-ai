from google import genai
from google.genai import types

from utils.config import get_settings


class GeminiClientService(genai.Client):
    def __init__(self) -> None:
        settings = get_settings()
        super().__init__(api_key=settings.gemini_api_key)
        self.embedding_model_name = settings.gemini_embedding_model_name
        self.output_dimensions = settings.gemini_output_dimensions

    def _build_embedding_model_config(self) -> types.EmbedContentConfig:
        return types.EmbedContentConfig(
            output_dimensionality=self.output_dimensions,
        )

    def is_healthy(self) -> bool:
        models_list_response = self.models.list(
            config=types.ListModelsConfig(page_size=1)
        )

        return (
            models_list_response.sdk_http_response.headers is not None
            and models_list_response.config is not None
        )
