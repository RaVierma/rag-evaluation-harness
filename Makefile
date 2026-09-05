.PHONY: install test lint format check run compare regression

install:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

check:
	uv run ruff check .
	uv run pytest

run:
	uv run python examples/basic_evaluation.py

compare:
	uv run python examples/compare_versions.py

regression:
	uv run python examples/regression_check.py