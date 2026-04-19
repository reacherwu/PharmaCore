.PHONY: install dev test lint clean benchmark

install:
	pip install .

dev:
	pip install -e ".[apple,admet,protein]"
	pip install pytest pytest-cov ruff mypy

test:
	python -m pytest tests/ -v

lint:
	ruff check pharmacore/ tests/
	mypy pharmacore/

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +

benchmark:
	python -m pharmacore.core.device
