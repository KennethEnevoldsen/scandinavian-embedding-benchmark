from typing import List

import catalogue

from .model_interface import SebModel
from .tasks_interface import Task

models = catalogue.create("seb", "models")
tasks = catalogue.create("seb", "tasks")


def get_model(name: str) -> SebModel:
    """
    Fetches a model by name.

    Args:
        name: The name of the model.

    Returns:
        A model including metadata.
    """
    return models.get(name)()


def get_task(name: str) -> Task:
    """
    Fetches a task by name.

    Args:
        name: The name of the task.

    Returns:
        A task.
    """
    return tasks.get(name)()


def get_all_tasks() -> List[Task]:
    """
    Returns all tasks implemented in SEB.

    Returns:
        A list of all tasks in SEB.
    """
    return [get_task(task_name) for task_name in tasks.get_all()]


def get_all_models() -> List[SebModel]:
    """
    Get all the models implemented in SEB.

    Returns:
        A list of all models in SEB.
    """
    return [get_model(model_name) for model_name in models.get_all()]
