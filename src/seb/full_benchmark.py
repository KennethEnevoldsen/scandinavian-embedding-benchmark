"""
This is the specification for the full benchmark. Running the code here will reproduce the results.
"""


from seb.model_interface import SebModel

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
    raise_errors: bool = True,
) -> dict[str, list[BenchmarkResults]]:
    """
    Run the full SEB benchmark.
    """
    models: list[SebModel] = get_all_models()

    results = {}
    for subset, langs in BENCHMARKS.items():
        benchmark = Benchmark(languages=langs)
        bm_results = benchmark.evaluate_models(
            models=models,
            use_cache=use_cache,
            raise_errors=raise_errors,
        )

        results[subset] = bm_results

    return results
