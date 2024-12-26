add-uv:
	@echo "--- 🚀 Installing UV ---"	
	curl -LsSf https://astral.sh/uv/install.sh | sh
	# windows:
	# powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

install:
	@echo "--- 🚀 Installing project ---"
	uv sync --extra dev --extra docs --extra tests

static-type-check:
	@echo "--- 🔍 Running static type check ---"
	pyright src/

lint:
	@echo "--- 🧹 Running linters ---"
	uv run ruff format . 						            # running ruff formatting
	uv run ruff check **/*.py --fix						# running ruff linting

lint-check:
	@echo "--- 🧹 Check is project is linted ---"
	uv run ruff format . --check						    # running ruff formatting
	uv run ruff check **/*.py 						        # running ruff linting

test:
	@echo "--- 🧪 Running tests ---"
	uv run pytest tests/

pr:
	@echo "--- 🚀 Running PR checks ---"
	make lint
	make static-type-check
	make test
	python src/scripts/check_benchmark_is_up_to_date.py
	@echo "Ready to make a PR"

update-table-in-docs:
	@echo "--- 🔄 Updating table in docs ---"
	python src/scripts/create_desc_stats.py

build-docs:
	@echo "--- 📚 Building docs ---"
	@echo "Builds the docs and puts them in the 'site' folder"
	@echo "You might need to also update the table with the desc. stats you can do this by running 'make update-table-in-docs'"
	uv run mkdocs build

view-docs:
	@echo "--- 👀 Viewing docs ---"
	uv run mkdocs serve
	
update-from-template:
	@echo "--- 🔄 Updating from template ---"
	@echo "This will update the project from the template, make sure to resolve any .rej files"
	cruft update --skip-apply-ask

run-benchmark:
	# HF API key to access the required datasets
	hf_api_key=$(cat hf_api_key.txt)
	export HF_TOKEN=hf_api_key

	# additionally this expect that API keys required for specific models are set as env variables

	# run benchmark
	seb run

check-benchmark-is-up-to-date:
	@echo "--- 🔄 Checking benchmark is up to date ---"

	uv run python src/scripts/check_benchmark_is_up_to_date.py

