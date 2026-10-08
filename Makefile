.PHONY: install env lint test test-all data repro reference docs docs-build clean

install:
	pip install -e ".[dev]"

env:
	conda env create -f environment.yaml

lint:
	pre-commit run --all-files

test:
	pytest -m "not slow and not repro"

test-all:
	pytest

data:
	adapt-decomp data get neuromotion-data

repro:
	pytest tests/reproducibility -m repro

reference:
	python tests/reproducibility/make_reference.py

docs:
	mkdocs serve

docs-build:
	mkdocs build --strict

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
