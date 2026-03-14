FROM python:3.12-slim AS builder

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
RUN pip install uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src

FROM python:3.12-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
RUN pip install uv

COPY --from=builder /app/.venv /app/.venv
COPY src ./src
COPY pyproject.toml uv.lock ./

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    SERVER_PORT=8001 \
    ENVIRONMENT=production

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
