from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


def build_openai_compatible_model(base_url: str, api_key: str, model_name: str, timeout: float) -> OpenAIChatModel:
    """Build a chat model for any provider that implements the OpenAI Chat Completions API (OpenAI, OpenRouter, Ollama, vLLM, LiteLLM)."""
    client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
    return OpenAIChatModel(model_name, provider=OpenAIProvider(openai_client=client))
