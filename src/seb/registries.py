from typing import List

import catalogue

from .model_interface import SebModel
from .tasks_interface import Task

models = catalogue.create("seb", "models")
tasks = catalogue.create("seb", "tasks")


def get_model(name: str) -> SebModel:
    return models.get(name)()


def get_task(name: str) -> Task:
    return tasks.get(name)()


def get_all_tasks() -> List[Task]:
    return [get_task(task_name) for task_name in tasks.get_all()]


def get_all_models() -> List[SebModel]:
    return [get_model(model_name) for model_name in models.get_all()]
