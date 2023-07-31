"""
The test specifications for the benchmark.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import pytest
from dummy_model import create_test_model  # noqa: F401
from dummy_task import create_test_task  # noqa: F401
from test_tasks import all_tasks_names

import seb


@pytest.mark.parametrize(
    "model_names, languages, tasks",
    [
        (
            ["sentence-transformers/all-MiniLM-L6-v2"],
            ["test task", "DKHate"],
            None,
        ),
        (
            [
                "test_model",
                "sentence-transformers/all-MiniLM-L6-v2",
            ],
            ["tes task", "test encode task"],
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

    if tasks is None:
        tasks = all_tasks_names  # to avoid test tasks that raise errors

    benchmark: seb.Benchmark = seb.Benchmark(
        languages=languages,
        tasks=tasks,
    )
    bm_results: List[seb.BenchmarkResults] = benchmark.evaluate_models(
        models=models, use_cache=False
    )

    assert len(bm_results) == len(models)

    n_tasks = len(benchmark.tasks)
    for bm_result in bm_results:
        assert len(bm_result) == n_tasks
        for task_result in bm_result:
            ensure_correct_task_result(task_result)


def ensure_correct_task_result(task_result: Union[seb.TaskResult, seb.TaskError]):
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
            ["test encode task"],
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


def set_cache_dir():
    new_cache_dir = Path(__file__).parent / "tmp_cache"
    os.environ["SEB_CACHE_DIR"] = str(new_cache_dir)
    new_cache_dir.mkdir(exist_ok=True)


def test_benchmark_skip_on_error_raised():
    """
    Test that the benchmark skips a model if an error is raised.
    """
    set_cache_dir()
    model = seb.get_model("test_model")
    benchmark: seb.Benchmark = seb.Benchmark(
        languages=None,
        tasks=["test raise error task"],
    )

    bm_result: seb.BenchmarkResults = benchmark.evaluate_model(
        model, use_cache=False, raise_errors=False
    )

    assert len(bm_result) == 1
    task_result = bm_result[0]
    assert task_result.task_name == "test raise error task"
    assert isinstance(task_result, seb.TaskError)

    # test that the benchmark raises an error if raise_errors is True
    with pytest.raises(ValueError):
        benchmark.evaluate_model(model, use_cache=False, raise_errors=True)
