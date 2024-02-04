from .benchmark import Benchmark, run_task
from .full_benchmark import run_benchmark, run_speed_benchmark
from .registries import (
    get_all_models,
    get_all_tasks,
    get_model,
    get_task,
    models,
    tasks,
)

from .interfaces.model import Encoder, LazyLoadEncoder, ModelMeta, SebModel
from .interfaces.task import Task
from .registered_models import *  # import all SEB models
from .registered_tasks import *  # import all SEB tasks
from .result_dataclasses import BenchmarkResults, TaskError, TaskResult
