"""Unit tests for core.dependencies app.state accessors."""

from unittest.mock import MagicMock

from core.dependencies import get_chat_model, get_database, get_redis


def test_new_dependencies_read_app_state() -> None:
    request = MagicMock()
    request.app.state.database = database = object()
    request.app.state.redis = redis = object()
    request.app.state.chat_model = chat_model = object()

    assert get_database(request) is database
    assert get_redis(request) is redis
    assert get_chat_model(request) is chat_model
