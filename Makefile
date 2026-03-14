.PHONY: run install format build docker-run

run:
	PYTHONPATH=src uv run python src/main.py

install:
	uv sync --extra dev

format:
	uv run ruff format src
	uv run ruff check src --fix

build:
	docker build -t dexbooru-ai:latest .

run-build: build
	docker run -d --name dexbooru-ai --rm -p 8001:8001 --env-file .env dexbooru-ai:latest