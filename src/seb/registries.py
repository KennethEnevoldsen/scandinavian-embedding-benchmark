import catalogue

from .model_interface import SebModel
from .tasks import Task

models = catalogue.create("seb", "models")
tasks = catalogue.create("seb", "tasks")


def get_model(name: str) -> SebModel:
    return models.get(name)

def get_task(name: str) -> Task:
    return tasks.get(name)