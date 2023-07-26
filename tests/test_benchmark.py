"""
The test specifications for the benchmark.
"""

from datetime import datetime
from typing import List, Optional

import pytest
from test_model import create_test_model  # noqa: F401

import seb


@pytest.mark.parametrize(
    "model_names, languages, tasks",
    [
        (
            ["sentence-transformers/all-MiniLM-L6-v2"],
            ["LCC", "DKHate"],
            None,
        ),
        (["sentence-transformers/all-MiniLM-L6-v2"], None, None),
        (
            [
                "test_model",
                "sentence-transformers/all-MiniLM-L6-v2",
            ],
            ["da"],
            None,
        ),
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
    bm_results: List[seb.BenchmarkResults] = benchmark.evaluate(
        models=models, use_cache=False
    )

    assert len(bm_results) == len(models)

    n_tasks = len(benchmark.tasks)
    for bm_result in bm_results:
        assert len(bm_result) == n_tasks
        for task_result in bm_result:
            ensure_correct_task_result(task_result)


def ensure_correct_task_result(task_result: seb.TaskResult):
    """
    Ensure that the task result is correct.
    """

    assert isinstance(task_result, seb.TaskResult)
    assert isinstance(task_result.task_name, str)
    assert isinstance(task_result.languages, list)

    main_score = task_result.get_main_score()
    assert isinstance(main_score, float)


@pytest.mark.parametrize(
    "model_name, languages, tasks",
    [
        (
            "sentence-transformers/all-MiniLM-L6-v2",
            None,
            ["DKHate"],
        ),
    ],
)
def test_cache_dir_is_reused(
    model_name: str,
    languages: Optional[List[str]],
    tasks: Optional[List[str]],
):
    """
    Check that the cache dir is reused.
    """
    model = seb.get_model(model_name)
    benchmark: seb.Benchmark = seb.Benchmark(
        languages=languages,
        tasks=tasks,
    )

    before_run = datetime.now()
    bm_result: seb.BenchmarkResults = benchmark.evaluate_model(model, use_cache=False)
    after_run = datetime.now()

    assert len(bm_result) == 1
    task_result_1 = bm_result[0]
    not_used_cache = before_run < task_result_1.time_of_run < after_run
    assert not_used_cache

    bm_result: seb.BenchmarkResults = benchmark.evaluate_model(
        model=model, use_cache=True
    )

    assert isinstance(bm_result, seb.BenchmarkResults)

    task_result_2 = bm_result[0]

    used_cache = task_result_1.time_of_run == task_result_2.time_of_run
    assert used_cache
    assert task_result_1 == task_result_2, "The two task results should be equal"
