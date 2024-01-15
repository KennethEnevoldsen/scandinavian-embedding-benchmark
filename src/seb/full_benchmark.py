"""
This is the specification for the full benchmark. Running the code here will reproduce the results.
"""


from pathlib import Path
from typing import Optional

from seb.model_interface import EmbeddingModel

from .benchmark import Benchmark
from .registries import get_all_models
from .result_dataclasses import BenchmarkResults

BENCHMARKS = {
    "Mainland Scandinavian": ["da", "sv", "nn", "nb"],
    "Danish": ["da"],
    "Norwegian": ["nn", "nb"],
    "Swedish": ["sv"],
}


def run_benchmark(
    use_cache: bool = True,
    run_models: bool = True,
    raise_errors: bool = True,
    cache_dir: Optional[Path] = None,
) -> dict[str, list[BenchmarkResults]]:
    """
    Run the full SEB benchmark.
    """
    models: list[EmbeddingModel] = get_all_models()

    results = {}
    for subset, langs in BENCHMARKS.items():
        benchmark = Benchmark(languages=langs)
        bm_results = benchmark.evaluate_models(
            models=models,
            use_cache=use_cache,
            run_model=run_models,
            raise_errors=raise_errors,
            cache_dir=cache_dir,
        )

        results[subset] = bm_results

    return results
