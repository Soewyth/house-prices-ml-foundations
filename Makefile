.PHONY: setup test lint up down # define commands 

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
	docker compose up --build -d

down:
	docker compose down
