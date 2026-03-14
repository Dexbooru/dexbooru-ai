# dexbooru-ai

A Python microservice that powers AI features for [Dexbooru](https://github.com/Dexbooru/dexbooru-web): **image similarity**, **post similarity with text**, vector search, and RAG (retrieval-augmented generation) over the image board’s posts and metadata.

## What it does

- **Image similarity** — Embeds images and finds visually similar posts via nearest-neighbour search in a vector space.
- **Post similarity with text** — Embeds both images and text (tags, captions, descriptions) so you can search by text (“red hair, school uniform”) and get posts that match semantically, not just by keyword.
- **Vector search & RAG** — Stores post and image embeddings in a vector database and supports RAG-style retrieval for downstream ML or search features.

Events (e.g. new posts) are consumed from a message broker; the service embeds content, writes vectors, and keeps the index in sync with the board.

## Tech stack

| Layer           | Technology                    |
| --------------- | ----------------------------- |
| Language        | **Python** (3.12+)            |
| API             | **FastAPI**                   |
| Vector database | **Qdrant**                    |
| Message broker  | **RabbitMQ** (exchange-based) |

- **Python** — Runtime and tooling (uv, pytest, ruff).
- **FastAPI** — HTTP API and lifespan (start/stop consumers).
- **Qdrant** — Vector store for embeddings; used for similarity search and RAG.
- **RabbitMQ** — Message broker; the service consumes from a shared **exchange** (e.g. `ai_events`), with per-queue consumers for different event types (e.g. new posts).

## Prerequisites

- **Python 3.12+**
- **uv** — [Install uv](https://docs.astral.sh/uv/getting-started/installation/)
- **Docker** (optional) — For Qdrant and, if you like, RabbitMQ
- **RabbitMQ** — Running locally or in Docker; the app connects via `AMQP_URL`
- **Qdrant** — Running locally or via the project’s `docker-compose` (see below)

## Setup

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd dexbooru-ai
make install
```

This runs `uv sync --extra dev` (installs main and dev dependencies, including pytest and pre-commit).

### 2. Environment

Copy the example env and set at least the required variables:

```bash
cp .env.example .env
# Edit .env with your AMQP_URL, SERVER_PORT, etc.
```

Main options (see `.env.example`):

- `SERVER_NAME`, `SERVER_PORT` — API server
- `AMQP_URL` — RabbitMQ URL (e.g. `amqp://guest:guest@localhost:5672`)
- `PRIMARY_EXCHANGE_NAME` — Exchange name consumers use (default: `ai_events`)
- `LOG_LEVEL` — e.g. `INFO`, `DEBUG`

### 3. Start Qdrant (vector database)

Using the project’s Docker Compose (Qdrant only):

```bash
make init-qdrant
```

This runs `docker compose up -d qdrant`; Qdrant is available at `http://localhost:6333`.

### 4. RabbitMQ

Ensure RabbitMQ is running and reachable at the host/port in `AMQP_URL`. The app does not start RabbitMQ for you; run it yourself (e.g. `docker run -d -p 5672:5672 -p 15672:15672 rabbitmq:3-management`) or use an existing instance.

## Running the project

From the repo root:

| Command        | Description                                                    |
| -------------- | -------------------------------------------------------------- |
| `make run`     | Start the FastAPI app (consumers start with the app lifespan). |
| `make test`    | Run the test suite (`pytest test/`).                           |
| `make format`  | Format and fix lint with Ruff.                                 |
| `make install` | Install dependencies (`uv sync --extra dev`).                  |

### Run the API

```bash
make run
```

This sets `PYTHONPATH=src` and runs `python src/main.py`. The server listens on the port from `SERVER_PORT` (default 8001 in `.env.example`). On startup, the app runs the lifespan: it starts all consumers (each with its own RabbitMQ connection and channel) and logs “Starting N consumer(s)” and “Started consumer: …”.

### Run tests

```bash
make test
```

Runs `pytest test/` with `PYTHONPATH=src`. Tests are also wired into the pre-commit hook so they must pass before a commit.

### Docker build and run

```bash
make build          # Build image dexbooru-ai:latest
make run-build      # Run container (port 8001, env from .env)
```

## Makefile reference

| Target             | Command / behaviour                                  |
| ------------------ | ---------------------------------------------------- |
| `make install`     | `uv sync --extra dev`                                |
| `make run`         | `PYTHONPATH=src uv run python src/main.py`           |
| `make test`        | `PYTHONPATH=src uv run pytest test/ -v`              |
| `make format`      | `ruff format src` and `ruff check src --fix`         |
| `make build`       | `docker build -t dexbooru-ai:latest .`               |
| `make run-build`   | Build image then `docker run` with `.env`, port 8001 |
| `make init-qdrant` | `docker compose up -d qdrant`                        |

## License

See repository license file.
