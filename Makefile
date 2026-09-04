PYTHON ?= python
VENV   ?= .venv

ifeq ($(OS),Windows_NT)
    BIN := $(VENV)/Scripts
else
    BIN := $(VENV)/bin
endif

.PHONY: help install train test lint format serve app score drift registry docker-build docker-up clean

help:
	@echo "install       create venv + install dev deps"
	@echo "train         train, calibrate and register a model"
	@echo "test          run the pytest suite"
	@echo "lint          ruff check"
	@echo "format        ruff format + fix"
	@echo "serve         run the REST API (uvicorn, reload)"
	@echo "app           run the Streamlit dashboard"
	@echo "score IN= OUT= score a CSV"
	@echo "drift  IN=    PSI drift report for a CSV"
	@echo "registry      list registered model versions"
	@echo "docker-build  build the container image"
	@echo "docker-up     docker compose up (API + dashboard)"
	@echo "clean         remove caches and generated models"

install:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install -r requirements-dev.txt

train:
	$(BIN)/python -m churn train

test:
	$(BIN)/python -m pytest -q

lint:
	$(BIN)/python -m ruff check .

format:
	$(BIN)/python -m ruff format .
	$(BIN)/python -m ruff check . --fix

serve:
	$(BIN)/python -m churn serve --reload

app:
	$(BIN)/python -m streamlit run app.py

score:
	$(BIN)/python -m churn score --input "$(IN)" --output "$(OUT)"

drift:
	$(BIN)/python -m churn drift --current "$(IN)"

registry:
	$(BIN)/python -m churn registry list

docker-build:
	docker compose build

docker-up:
	docker compose up

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ logs
	rm -rf models/registry
