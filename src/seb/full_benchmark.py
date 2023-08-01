"""
This is the specification for the full benchmark. Running the code here will reproduce the results.
"""

from typing import List

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


def run_benchmark(use_cache: bool = True) -> dict[str, List[BenchmarkResults]]:
    """
    Run the full SEB benchmark.
    """
    models: List[SebModel] = get_all_models()

    results = {}
    for subset, langs in BENCHMARKS.items():
        benchmark = Benchmark(languages=langs)
        bm_results = benchmark.evaluate_models(
            models=models, use_cache=use_cache, raise_errors=True
        )

        results[subset] = bm_results

    return results
