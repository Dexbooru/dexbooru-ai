"""Unit tests for core.llm."""

from unittest.mock import patch

from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIChatModel

from core.llm import build_openai_compatible_model


class TestBuildOpenAICompatibleModel:
    def test_passes_settings_through(self) -> None:
        model = build_openai_compatible_model(
            base_url="http://localhost:11434/v1",
            api_key="secret-key",
            model_name="llama3.2",
            timeout=42,
        )

        assert isinstance(model, OpenAIChatModel)
        assert model.model_name == "llama3.2"
        client = model.client
        assert isinstance(client, AsyncOpenAI)
        assert str(client.base_url).rstrip("/") == "http://localhost:11434/v1"
        assert client.api_key == "secret-key"
        assert client.timeout == 42

    def test_constructs_client_with_expected_arguments(self) -> None:
        with patch("core.llm.AsyncOpenAI", wraps=AsyncOpenAI) as async_openai:
            build_openai_compatible_model("https://openrouter.ai/api/v1", "key", "openai/gpt-4o-mini", 30)

        async_openai.assert_called_once_with(base_url="https://openrouter.ai/api/v1", api_key="key", timeout=30)
