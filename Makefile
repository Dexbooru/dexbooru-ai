.PHONY: run install format build docker-run init-qdrant test test\:coverage

run:
	PYTHONPATH=src uv run python src/main.py

install:
	uv sync --extra dev

test:
	PYTHONPATH=src uv run pytest test/ -v

test\:coverage:
	./scripts/test-coverage.sh

format:
	uv run ruff format src test
	uv run ruff check src test --fix

build:
	docker build -t dexbooru-ai:latest .

run-build: build
	docker run -d --name dexbooru-ai --rm -p 8001:8001 --env-file .env dexbooru-ai:latest

init-qdrant:
	docker compose up -d qdrant