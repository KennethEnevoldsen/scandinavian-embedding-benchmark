from .benchmark import Benchmark
from .full_benchmark import run_benchmark
from .model_interface import ModelInterface, ModelMeta, SebModel
from .registries import (
    get_all_models,
    get_all_tasks,
    get_model,
    get_task,
    models,
    tasks,
)
from .result_dataclasses import BenchmarkResults, TaskError, TaskResult
from .seb_models import *  # import all SEB models
from .seb_tasks import *  # import all SEB tasks
from .tasks_interface import Task
