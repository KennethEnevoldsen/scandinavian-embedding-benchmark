from seb.cli import run_benchmark_cli


def test_cli_integration():
    """Runs all sorts of models on a small task to see if they can run without breaking.
    Cache is ignored so that the models are actually run.
    """
    models = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "intfloat/e5-small",
        "translate-e5-small",
        "fasttext-cc-da-300",
    ]
    tasks = ["LCC"]
    run_benchmark_cli(models=models, tasks=tasks, ignore_cache=True)
