.PHONY: install dev test lint fmt run clean docker

install:
	pip install -e .

dev:
	pip install -e ".[all]"

test:
	python -m pytest tests/ -v --tb=short

test-cov:
	python -m pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/

fmt:
	ruff format src/ tests/

run:
	streamlit run src/viz/app.py

demo:
	python examples/demo.py

cli-infinite:
	python -m src.cli infinite --delta1 0.9 --delta2 0.8 -v

cli-finite:
	python -m src.cli finite --delta1 0.9 --delta2 0.8 --rounds 10 -v

cli-nash:
	python -m src.cli nash --surplus 1.0 --alpha 0.5

cli-convergence:
	python -m src.cli convergence

cli-multi:
	python -m src.cli multi --v1 "0.8,0.3,0.6" --v2 "0.3,0.7,0.5"

docker:
	docker build -t rubinstein-bargaining .
	docker run -p 8501:8501 rubinstein-bargaining

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name *.egg-info -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ .coverage htmlcov/
