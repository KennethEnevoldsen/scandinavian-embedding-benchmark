install:
	pip install -e ".[dev, docs, openai, cohere, tests]" 

static-type-check:
	pyright src/

lint:
	pre-commit run --all-files

test:
	pytest tests/

pr:
	make lint
	make static-type-check
	make test
	echo "Ready to make a PR"

docs-serve:
	mkdocs serve

update-benchmark:
	python docs/run_benchmark.py --data-wrapper-api-token MISSING
