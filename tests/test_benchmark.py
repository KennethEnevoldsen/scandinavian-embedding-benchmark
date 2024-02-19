"""
The test specifications for the benchmark.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import pytest
import seb

from .dummy_model import create_test_model
from .dummy_task import (
    create_test_encode_task,
    create_test_raise_error_task,
    create_test_task,
)
from .test_tasks import all_tasks_names


@pytest.mark.parametrize(
    ("model_names", "languages", "tasks"),
    [
        (
            ["all-MiniLM-L6-v2"],
            None,
            [create_test_task()],
        ),
        (
            [
                "test_model",
                "all-MiniLM-L6-v2",
            ],
            None,
            [create_test_task(), create_test_encode_task()],
        ),
    ],
)
def test_run_benchmark(
    model_names: list[str],
    languages: Optional[list[str]],
    tasks: Optional[Union[list[str], list[seb.Task]]],
    tmp_path: Path,
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
    bm_results: list[seb.BenchmarkResults] = benchmark.evaluate_models(models=models, use_cache=False, cache_dir=tmp_path)

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
    ("model_name", "languages", "tasks"),
    [
        (
            "all-MiniLM-L6-v2",
            None,
            [create_test_encode_task()],
        ),
    ],
)
def test_cache_dir_is_reused(
    model_name: str,
    languages: Optional[list[str]],
    tasks: Optional[Union[list[str], list[seb.Task]]],
    tmp_path: Path,
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
    bm_result: seb.BenchmarkResults = benchmark.evaluate_model(model, use_cache=False, cache_dir=tmp_path)
    after_run = datetime.now()

    assert len(bm_result) == 1
    task_result_1 = bm_result[0]
    not_used_cache = before_run < task_result_1.time_of_run < after_run
    assert not_used_cache

    bm_result: seb.BenchmarkResults = benchmark.evaluate_model(
        model=model,
        use_cache=True,
        cache_dir=tmp_path,
    )

    assert isinstance(bm_result, seb.BenchmarkResults)

    task_result_2 = bm_result[0]

    used_cache = task_result_1.time_of_run == task_result_2.time_of_run
    assert used_cache
    assert task_result_1 == task_result_2, "The two task results should be equal"


def test_benchmark_skip_on_error_raised(tmp_path: Path):
    """
    Test that the benchmark skips a model if an error is raised.
    """
    task = create_test_raise_error_task()
    model = seb.get_model("test_model")
    benchmark: seb.Benchmark = seb.Benchmark(
        languages=None,
        tasks=[task],
    )

    bm_result: seb.BenchmarkResults = benchmark.evaluate_model(
        model,
        use_cache=False,
        raise_errors=False,
        cache_dir=tmp_path,
    )

    assert len(bm_result) == 1
    task_result = bm_result[0]
    assert task_result.task_name == "test raise error task"
    assert isinstance(task_result, seb.TaskError)

    # test that the benchmark raises an error if raise_errors is True
    with pytest.raises(ValueError):  # noqa: PT011
        benchmark.evaluate_model(model, use_cache=False, raise_errors=True, cache_dir=tmp_path)


@pytest.mark.parametrize("languages", [None, ["nb", "nn"], ["sv", "nb", "nn"], ["sv", "nb", "nn", "da"]])
@pytest.mark.parametrize("tasks", [None, ["DKHate", "ScaLA"]])
def test_benchmark_init(languages: Optional[list[str]], tasks: Optional[list[str]]):
    benchmark = seb.Benchmark(languages=languages, tasks=tasks)

    if languages:
        lang_set = set(languages)
        for task in benchmark.tasks:
            assert any(lang in lang_set for lang in task.languages), "the benchmark should not include tasks with languages not specified"
    if tasks:
        tasks_names = {t.name for t in benchmark.tasks}
        assert tasks_names.issubset(set(tasks)), "The tasks should be the same or less than the tasks specified"
