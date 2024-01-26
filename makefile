install:
	@echo "--- 🚀 Installing project ---"
	pip install -e ".[dev, docs, openai, cohere, tests, mistral, fasttext]" 

static-type-check:
	@echo "--- 🔍 Running static type check ---"
	pyright src/

lint:
	@echo "--- 🧹 Running linters ---"
	ruff format . 								# running ruff formatting
	ruff **/*.py --fix 						    # running ruff linting

lint-check:
	@echo "--- 🧹 Check is project is linted ---"
	ruff format . --check						# running ruff formatting
	ruff **/*.py 						        # running ruff linting

test:
	@echo "--- 🧪 Running tests ---"
	pytest tests/

pr:
	@echo "--- 🚀 Running PR checks ---"
	make lint
	make static-type-check
	make test
	@echo "Ready to make a PR"

update-table-in-docs:
	@echo "--- 🔄 Updating table in docs ---"
	python src/scripts/create_desc_stats.py

build-docs:
	@echo "--- 📚 Building docs ---"
	@echo "Builds the docs and puts them in the 'site' folder"
	@echo "You might need to also update the table with the desc. stats you can do this by running 'make update-table-in-docs'"
	mkdocs build

view-docs:
	@echo "--- 👀 Viewing docs ---"
	mkdocs serve
	
update-from-template:
	@echo "--- 🔄 Updating from template ---"
	@echo "This will update the project from the template, make sure to resolve any .rej files"
	cruft update --skip-apply-ask

run-benchmark:
	# set environment variables
	hf_api_key=$(cat hf_api_key.txt)
	export HF_TOKEN=hf_api_key

	# run benchmark
	datawrapper_api_key=$(cat datawrapper_api_key.txt)
	python docs/run_benchmark.py --data-wrapper-api-token $datawrapper_api_key


check-benchmark-is-up-to-date:
	@echo "--- 🔄 Checking benchmark is up to date ---"

	python src/scripts/check_benchmark_is_up_to_date.py

