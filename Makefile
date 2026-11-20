# Shortcuts for the space mission success prediction project.
# Usage: make <target>  (override the interpreter with `make PY=python3.11 ...`)

PY ?= python3
PIP ?= pip

.PHONY: setup test demo clean

setup: ## Install dependencies
	$(PIP) install -r requirements.txt

test: ## Run the pytest suite
	$(PY) -m pytest -q

demo: ## Run the pipeline demo on the real dataset
	$(PY) examples/run_pipeline_demo.py

clean: ## Remove caches and bytecode
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .mypy_cache
