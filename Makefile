.PHONY: setup test lint up down logs

setup:
	python -m venv .venv && \
	. .venv/bin/activate && \
	pip install --upgrade pip && \
	pip install -r requirements.txt && \
	pip install -e .

test:
	python -m pytest src/house_prices_ml_foundations/tests/ -q

lint:
	ruff check . && ruff format --check .

up:
	DOCKER_UID=$(shell id -u) DOCKER_GID=$(shell id -g) docker compose up --build -d
	@echo ""
	@echo "MLflow UI  : http://localhost:5000"
	@echo "API docs   : http://localhost:8000/docs"
	@echo ""
	@echo "Run 'make up' before launching any training script."

down:
	docker compose down

logs:
	docker compose logs -f
