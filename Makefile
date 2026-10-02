.PHONY: install env lint test test-all data repro reference clean

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
	python scripts/download_data.py get neuromotion-data

repro:
	pytest tests/reproducibility -m repro

reference:
	python tests/reproducibility/make_reference.py

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
