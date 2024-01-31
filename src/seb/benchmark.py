import logging
import os
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from tqdm import tqdm

from .interfaces.model import Encoder, SebModel
from .interfaces.task import Task
from .registries import get_all_tasks, get_task
from .result_dataclasses import BenchmarkResults, TaskError, TaskResult
from .warning_ignore_manager import WarningIgnoreContextManager

logger = logging.getLogger(__name__)


package_dir = Path(__file__).parent
CACHE_DIR = package_dir / "cache"


def get_cache_dir() -> Path:
    """
    Get the cache directory for SEB. Can be overridden by setting the environment
    variable SEB_CACHE_DIR.
    """
    cache_dir = os.environ.get("SEB_CACHE_DIR")
    if cache_dir is not None:
        return Path(cache_dir)
    return CACHE_DIR


def get_cache_path(task: Task, model: SebModel, cache_dir: Optional[Path] = None) -> Path:
    """
    Get the cache path for a task and model.
    """
    cache_path = cache_dir if cache_dir is not None else get_cache_dir()
    mdl_path_name = model.meta.get_path_name()
    task_path_name = task.name_to_path() + ".json"
    task_cache_path = cache_path / mdl_path_name / task_path_name
    return task_cache_path


def run_task(
    task: Task,
    model: SebModel,
    use_cache: bool,
    run_model: bool,
    raise_errors: bool,
    cache_dir: Optional[Path] = None,
) -> Union[TaskResult, TaskError]:
    """
    Tests a model on a task
    """
    if run_model is False and use_cache is False:
        raise ValueError("run_model and use_cache cannot both be False")

    if not raise_errors:
        try:
            return run_task(
                task=task,
                model=model,
                use_cache=use_cache,
                run_model=run_model,
                raise_errors=True,
                cache_dir=cache_dir,
            )
        except Exception as e:
            logger.error(f"Error when running {task.name} on {model.meta.name}: {e}")
            return TaskError(
                task_name=task.name,
                error=str(e),
                time_of_run=datetime.now(),
            )

    cache_path = get_cache_path(task, model, cache_dir)
    if cache_path.exists() and use_cache:
        logger.debug(f"Loading cached result for {model.meta.name} on {task.name}")
        task_result = TaskResult.from_disk(cache_path)
        return task_result

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if not run_model:
        raise ValueError(
            f"Cache for {model.meta.name} on {task.name} does not exist. " "Set run_model=True to run the model.",
        )
    with WarningIgnoreContextManager():
        task_result = task.evaluate(model.encoder)
    task_result.to_disk(cache_path)
    return task_result


class Benchmark:
    """
    Benchmark is the main orchestrator of the SEB benchmark.
    """

    def __init__(
        self,
        languages: Optional[list[str]] = None,
        tasks: Optional[Union[Iterable[str], Iterable[Task]]] = None,
    ) -> None:
        """
        Initialize the benchmark.

        Args:
            languages: A list of languages to run the benchmark on. If None, all languages are used.
            tasks: The tasks to run the benchmark on. If None, all tasks are used. Can either be specified as strings or as Task objects.
        """
        self.languages = languages

        self.tasks = self.get_tasks(tasks, languages)
        self.task_names = [task.name for task in self.tasks]

    @staticmethod
    def get_tasks(
        tasks: Optional[Union[Iterable[str], Iterable[Task]]],
        languages: Optional[list[str]],
    ) -> list[Task]:
        """
        Get the tasks for the benchmark.

        Returns:
            A list of tasks.
        """
        _tasks = []

        if tasks is None:
            _tasks = get_all_tasks()
        else:
            for task in tasks:
                if isinstance(task, str):
                    _tasks.append(get_task(task))
                elif isinstance(task, Task):
                    _tasks.append(task)
                else:
                    raise ValueError(f"Invalid task type: {type(task)}")

        if languages is not None:
            langs = set(languages)
            _tasks = [task for task in _tasks if set(task.languages) & langs]

        return _tasks

    def evaluate_model(
        self,
        model: SebModel,
        use_cache: bool = True,
        run_model: bool = True,
        raise_errors: bool = True,
        cache_dir: Optional[Path] = None,
        verbose: bool = True,
    ) -> BenchmarkResults:
        """
        Evaluate a model on the benchmark.

        Args:
            model: The model to evaluate.
            use_cache: Whether to use the cache.
            run_model: Whether to run the model if the cache is not present.
            raise_errors: Whether to raise errors.
            cache_dir: The cache directory to use. If None, the default cache directory is used.
            verbose: Whether to show a progress bar.

        Returns:
            The results of the benchmark.
        """
        task_results = []
        pbar = tqdm(
            self.tasks,
            position=1,
            desc=f"Running {model.meta.name}",
            leave=False,
            disable=not verbose,
        )
        for task in pbar:
            pbar.set_description(f"Running {model.meta.name} on {task.name}")
            task_result = run_task(
                task,
                model,
                use_cache=use_cache,
                run_model=run_model,
                raise_errors=raise_errors,
                cache_dir=cache_dir,
            )
            task_results.append(task_result)

        return BenchmarkResults(meta=model.meta, task_results=task_results)

    def evaluate_models(
        self,
        models: list[SebModel],
        use_cache: bool = True,
        run_model: bool = True,
        raise_errors: bool = True,
        cache_dir: Optional[Path] = None,
        verbose: bool = True,
    ) -> list[BenchmarkResults]:
        """
        Evaluate a list of models on the benchmark.

        Args:
            models: The models to evaluate.
            use_cache: Whether to use the cache.
            run_model: Whether to run the model if the cache is not present.
            raise_errors: Whether to raise errors.
            cache_dir: The cache directory to use. If None, the default cache directory is used.
            verbose: Whether to show a progress bar.

        Returns:
            The results of the benchmark, once for each model.
        """
        results = []
        pbar = tqdm(
            models,
            position=0,
            desc="Running Benchmark",
            leave=True,
            disable=not verbose,
        )

        for model in pbar:
            pbar.set_description(f"Running {model.meta.name}")
            results.append(
                self.evaluate_model(
                    model,
                    use_cache=use_cache,
                    run_model=run_model,
                    raise_errors=raise_errors,
                    cache_dir=cache_dir,
                    verbose=verbose,
                ),
            )
        return results
