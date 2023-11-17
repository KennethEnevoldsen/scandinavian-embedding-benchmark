install:
	pip install -e ".[dev, docs, openai, cohere, tests]"

update_benchmark:
	python docs/run_benchmark.py --data-wrapper-api-token MISSING