from typing import Protocol, runtime_checkable

from ..result_dataclasses import TaskResult
from ..types import DescriptiveDatasetStats, Domain, Language, TaskType
from .model import Encoder


@runtime_checkable
class Task(Protocol):
    """
    A task is a specific evaluation task for a sentence embedding model.

    Attributes:
        name: The name of the task.
        main_score: The main score of the task.
        description: A description of the task.
        reference: A reference to the task.
        version: The version of the task.
        languages: The languages of the task.
        domain: The domains of the task. Should be one of the categories listed on https://universaldependencies.org
    """

    name: str
    main_score: str
    description: str
    reference: str
    version: str
    languages: list[Language]
    domain: list[Domain]
    task_type: TaskType

    def evaluate(self, model: Encoder) -> TaskResult:
        """
        Evaluates a Sentence Embedding Model on the task.

        Args:
            model: A model with the encode method implemented.

        Returns:
            A TaskResult object.
        """
        ...

    def get_descriptive_stats(self) -> DescriptiveDatasetStats:
        ...

    def name_to_path(self) -> str:
        """
        Convert a name to a path.
        """
        name = self.name.replace("/", "__").replace(" ", "_")
        return name
