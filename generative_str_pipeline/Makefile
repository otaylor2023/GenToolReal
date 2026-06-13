.PHONY: install test lint

install:
	pip install -e ".[dev]"

test:
	pytest -q tests/

lint:
	python -m compileall generative_str
