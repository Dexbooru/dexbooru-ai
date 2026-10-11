import json
from collections.abc import AsyncIterable
from typing import Any

from fastapi.responses import StreamingResponse

SSE_MEDIA_TYPE = "text/event-stream"
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
}


def format_sse_event(event: str, data: Any) -> str:
    """Encode one SSE event. ``data`` is JSON-encoded, so it always fits on a single ``data:`` line."""
    if "\n" in event or "\r" in event:
        raise ValueError("SSE event name must not contain line breaks")
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def create_sse_response(events: AsyncIterable[str]) -> StreamingResponse:
    return StreamingResponse(events, media_type=SSE_MEDIA_TYPE, headers=SSE_HEADERS)
