"""
This is the specification for the full benchmark. Running the code here will reproduce the results.
"""

from typing import List

from seb.model_interface import SebModel

from .benchmark import Benchmark
from .registries import get_all_models
from .results import BenchmarkResults


def run_seb(use_cache: bool = True) -> dict[str, BenchmarkResults]:
    """
    Run the full benchmark.
    """
    models: List[SebModel] = get_all_models()

    subsets = {
        "Full": ["da", "no", "sv", "nn", "nb"],
        "Danish": ["da"],
        "Norwegian": ["no", "nn", "nb"],
        "Swedish": ["sv"],
    }
    results = {}
    for subset, langs in subsets.items():
        benchmark = Benchmark(languages=langs)
        bm_results = benchmark.evaluate(models=models, use_cache=use_cache)

        results[subset] = bm_results

    return results
