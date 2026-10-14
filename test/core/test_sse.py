"""Unit tests for core.sse."""

import asyncio
import json
from collections.abc import AsyncIterator

import pytest
from fastapi.responses import StreamingResponse

from core.sse import create_sse_response, format_sse_event


class TestFormatSseEvent:
    def test_formats_event_and_compact_json(self) -> None:
        assert format_sse_event("token", {"text": "hi", "n": 1}) == 'event: token\ndata: {"text":"hi","n":1}\n\n'

    def test_multiline_data_stays_on_one_data_line(self) -> None:
        frame = format_sse_event("token", {"text": "line one\nline two\r\n\ndata: injected"})

        lines = frame.split("\n")
        assert lines[0] == "event: token"
        assert lines[1].startswith("data: ")
        assert lines[2:] == ["", ""]
        assert json.loads(lines[1].removeprefix("data: ")) == {"text": "line one\nline two\r\n\ndata: injected"}

    def test_non_object_data(self) -> None:
        assert format_sse_event("done", None) == "event: done\ndata: null\n\n"
        assert format_sse_event("text", "plain") == 'event: text\ndata: "plain"\n\n'

    def test_unicode_passed_through(self) -> None:
        assert format_sse_event("token", "ねこ") == 'event: token\ndata: "ねこ"\n\n'

    @pytest.mark.parametrize("event", ["bad\nevent", "bad\revent"])
    def test_rejects_event_name_with_line_break(self, event: str) -> None:
        with pytest.raises(ValueError):
            format_sse_event(event, {})


class TestCreateSseResponse:
    def test_media_type_headers_and_body(self) -> None:
        async def events() -> AsyncIterator[str]:
            yield format_sse_event("token", {"text": "a"})
            yield format_sse_event("done", {})

        response = create_sse_response(events())

        assert isinstance(response, StreamingResponse)
        assert response.media_type == "text/event-stream"
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["cache-control"] == "no-cache"
        assert response.headers["x-accel-buffering"] == "no"

        async def collect() -> str:
            chunks = [chunk async for chunk in response.body_iterator]
            return "".join(c if isinstance(c, str) else c.decode() for c in chunks)

        assert asyncio.run(collect()) == 'event: token\ndata: {"text":"a"}\n\nevent: done\ndata: {}\n\n'
