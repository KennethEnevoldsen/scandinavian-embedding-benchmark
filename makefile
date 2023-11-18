install:
	pip install -e ".[dev, docs, openai, cohere, tests]" 

update_benchmark:
	python docs/run_benchmark.py --data-wrapper-api-token MISSING

static_type_check:
	pyright src/

lint:
	pre-commit run --all-files

test:
	pytest tests/

make pr:
	make lint
	make static_type_check
	make test
	echo "Ready to make a PR"