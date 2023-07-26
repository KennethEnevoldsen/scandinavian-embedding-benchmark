"""
The test specifications for the benchmark.
"""

from typing import List, Optional

import pytest

import seb


@pytest.mark.parametrize(
    "model_names, languages, tasks",
    [
        (["Maltehb/aelaectra-danish-electra-small-cased"], ["da"], None),
        (
            ["sentence-transformers/all-mpnet-base-v2"],
            ["LccSentimentClassification", "DKHateClassification"],
            None,
        ),
        (["sentence-transformers/all-mpnet-base-v2"], None, None),
        (["sentence-transformers/all-mpnet-base-v2"], None, None),
    ],
)
def test_run_benchmark(
    model_names: List[str],
    languages: Optional[List[str]],
    tasks: Optional[List[str]],
):
    """
    Test that the benchmark runs without errors.
    """
    models = [seb.get_model(model_name) for model_name in model_names]

    benchmark: seb.Benchmark = seb.Benchmark(
        languages=languages,
        tasks=tasks,
    )
    bm_results: List[seb.benchmarkResults] = benchmark.evaluate(
        models=models, use_cache=False
    )

    for bm_result in bm_results:
        for task_result in bm_result:
            ensure_correct_task_result(task_result)


def ensure_correct_task_result(task_result: seb.TaskResult):
    """
    Ensure that the task result is correct.
    """

    assert isinstance(task_result, seb.TaskResult)
    assert isinstance(task_result.task_name, str)
    assert isinstance(task_result.languages, list)

    json = task_result.to_json()
    assert isinstance(json, dict)

    main_score = task_result.get_main_score()
    assert isinstance(main_score, float)


@pytest.mark.parametrize(
    "model_name, languages, tasks",
    [
        (
            "Maltehb/aelaectra-danish-electra-small-cased",
            None,
            ["DKHateClassification"],
        ),
    ],
)
def check_cache_dir_is_reused(
    model_name: str,
    languages: Optional[List[str]],
    tasks: Optional[List[str]],
):
    """
    Check that the cache dir is reused.
    """
    models = seb.get_model(model_name)
    benchmark: seb.Benchmark = seb.Benchmark(
        languages=languages,
        tasks=tasks,
    )

    bm_result: seb.BenchmarkResult = benchmark.evaluate_model(model, use_cache=False)

    assert len(bm_result) == 1
    task_result_1 = bm_result[0]
    assert bm_result.loaded_from_cache == False

    bm_result: seb.BenchmarkResult = benchmark.evaluate_model(
        models=models, use_cache=True
    )
    task_result_2 = bm_result[0]
    assert bm_result.loaded_from_cache == True

    assert task_result_1.time_of_run == task_result_2.time_of_run
    assert task_result_1 == task_result_2, "The two task results should be equal"
