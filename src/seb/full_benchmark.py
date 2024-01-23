"""
This is the specification for the full benchmark. Running the code here will reproduce the results.
"""


import logging
from pathlib import Path
from typing import Optional

from .benchmark import Benchmark
from .interfaces.model import EmbeddingModel
from .registered_tasks.speed import CPUSpeedTask, GPUSpeedTask
from .registries import get_all_models
from .result_dataclasses import BenchmarkResults

logger = logging.getLogger(__name__)

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


def run_speed_benchmark(
    use_cache: bool = True,
    run_models: bool = True,
    raise_errors: bool = True,
    cache_dir: Optional[Path] = None,
) -> dict[str, list[BenchmarkResults]]:
    """
    Run the speed benchmark.
    """
    models: list[EmbeddingModel] = get_all_models()
    tasks = [CPUSpeedTask(), GPUSpeedTask()]  # type: ignore

    if use_cache:
        logger.warn(
            "Running the speed benchmark with use_cache=True will load speed results from the cache, this might lead to incomparable results."
        )

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
