"""
This is the specification for the full benchmark. Running the code here will reproduce the results.
"""

import seb


def run_seb(use_cache: bool = True) -> dict[str, seb.BenchmarkResults]:
    """
    Run the full benchmark.
    """
    models = seb.models.get_all().values()

    subsets = {
        "Full": ["da", "no", "sv", "nn", "nb"],
        "Danish": ["da"],
        "Norwegian": ["no", "nn", "nb"],
        "Swedish": ["sv"],
    }
    results = {}
    for subset, langs in subsets.items():
        benchmark = seb.Benchmark(languages=langs)
        bm_results = benchmark.evaluate(models=models, use_cache=use_cache)

        results[subset] = bm_results

    return results
