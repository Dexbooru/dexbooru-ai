# syntax=docker/dockerfile:1
#
# Model artifacts are pulled from S3 at build time
# Required build args: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

FROM python:3.12-slim AS builder

WORKDIR /app

# Build-time dependencies
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
RUN pip install uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Source code
COPY src ./src

# Fetch ML artifacts; credentials stay in this stage only (not ENV in the runtime image).
FROM python:3.12-slim AS model-fetch

ARG MODEL_S3_BUCKET=dexbooru-prd-machine-learning-models
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_DEFAULT_REGION=us-east-1

RUN apt-get update \
    && apt-get install -y --no-install-recommends awscli ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /out
RUN mkdir -p models/metadata \
    && export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
    AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \
    AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION}" \
    && aws s3 cp "s3://${MODEL_S3_BUCKET}/models/danbooru_tag_rating_predictor.skops" models/ \
    && aws s3 cp \
    "s3://${MODEL_S3_BUCKET}/models/metadata/danbooru_tag_rating_predictor.meta.json" \
    models/metadata/


# Runtime image
FROM python:3.12-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
RUN pip install uv

COPY --from=builder /app/.venv /app/.venv
COPY --from=model-fetch /out/models /app/models
COPY src ./src
COPY pyproject.toml uv.lock ./

# uv-managed venvs omit pip; spacy download expects pip unless we bootstrap it.
RUN uv pip install --python /app/.venv/bin/python pip \
    && /app/.venv/bin/python -m spacy download en_core_web_md

# Environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    SERVER_PORT=8001 \
    ENVIRONMENT=production \
    DANBOORU_TAG_RATING_SKOPS_PATH=/app/models/danbooru_tag_rating_predictor.skops \
    SPACY_ENGLISH_MODEL=en_core_web_md

# Expose port and run the application
EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
