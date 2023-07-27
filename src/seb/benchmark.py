from pathlib import Path
from typing import List, Optional, Union

from .model_interface import SebModel
from .registries import get_all_tasks, get_task
from .result_dataclasses import BenchmarkResults, TaskError, TaskResult
from .tasks_interface import Task
from .utils import get_cache_dir, name_to_path


def get_cache_path(task: Task, model: SebModel) -> Path:
    """
    Get the cache path for a task and model.
    """
    cache_path = get_cache_dir()
    mdl_path_name = model.meta.get_path_name()
    task_path_name = name_to_path(task.name) + ".json"
    task_cache_path = cache_path / mdl_path_name / task_path_name
    return task_cache_path


def run_task(
    task: Task, model: SebModel, use_cache: bool, raise_errors: bool
) -> Union[TaskResult, TaskError]:
    """
    Tests a model on a task
    """

    if not raise_errors:
        try:
            return run_task(task, model, use_cache, raise_errors=True)
        except Exception as e:
            return TaskError(task_name=task.name, error=str(e))

    cache_path = get_cache_path(task, model)
    if cache_path.exists() and use_cache:
        task_result = TaskResult.from_disk(cache_path)
        return task_result

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    task_result = task.evaluate(model)
    task_result.to_disk(cache_path)
    return task_result


class Benchmark:
    def __init__(
        self,
        languages: Optional[List[str]] = None,
        tasks: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the benchmark.
        """
        self.languages = languages
        self.tasks_names = tasks
        self.tasks = self.get_tasks()

    def get_tasks(self) -> List[Task]:
        """
        Get the tasks for the benchmark.
        """
        tasks = []

        if self.tasks_names is not None:
            tasks: List[Task] = [get_task(task_name) for task_name in self.tasks_names]
        else:
            tasks: List[Task] = get_all_tasks()

        if self.languages is not None:
            langs = set(self.languages)
            tasks = [task for task in tasks if set(task.languages) & langs]

        return tasks

    def evaluate_model(
        self, model: SebModel, use_cache: bool = True, raise_errors: bool = True
    ) -> BenchmarkResults:
        """
        Evaluate a model on the benchmark.
        """
        tasks = self.get_tasks()
        task_results = []
        for task in tasks:
            task_result = run_task(task, model, use_cache, raise_errors)
            task_results.append(task_result)

        return BenchmarkResults(meta=model.meta, task_results=task_results)

    def evaluate_models(
        self, models: List[SebModel], use_cache: bool = True, raise_errors: bool = True
    ) -> List[BenchmarkResults]:
        """
        Evaluate a list of models on the benchmark.
        """
        results = []
        for model in models:
            results.append(
                self.evaluate_model(
                    model, use_cache=use_cache, raise_errors=raise_errors
                )
            )
        return results
