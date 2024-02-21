from typing import Literal, Protocol, TypedDict, runtime_checkable

import numpy as np

from seb.interfaces.language import Language

from ..result_dataclasses import TaskResult
from .model import Encoder

Domain = Literal[
    "social",
    "poetry",
    "wiki",
    "fiction",
    "non-fiction",
    "web",
    "legal",
    "news",
    "academic",
    "spoken",
    "reviews",
    "blog",
    "medical",
    "government",
    "bible",
]

TaskType = Literal["Classification", "Retrieval", "STS", "BitextMining", "Clustering", "Speed"]


class DescriptiveDatasetStats(TypedDict):
    mean_document_length: float
    std_document_length: float
    num_documents: int


@runtime_checkable
class Task(Protocol):
    """
    A task is a specific evaluation task for a sentence embedding model.

    Attributes:
        name: The name of the task.
        main_score: The main score of the task.
        reference: A reference to the task.
        version: The version of the task.
        languages: The languages of the task.
        domain: The domains of the task. Should be one of the categories listed on https://universaldependencies.org
        task_type: A list of task types, determines how the task is being evaluated. E.g. Classification.
        task_subtypes: a list of subtypes e.g. Sentiment Classification.
        description: A description of the task.
    """

    name: str
    main_score: str
    reference: str
    version: str
    languages: list[Language]
    domain: list[Domain]
    task_type: TaskType
    task_subtypes: list[str]
    description: str

    def evaluate(self, model: Encoder) -> TaskResult:
        """
        Evaluates a Sentence Embedding Model on the task.

        Args:
            model: A model with the encode method implemented.

        Returns:
            A TaskResult object.
        """
        ...

    def get_documents(self) -> list[str]:
        """
        Get the documents for the task.

        Returns:
            A list of strings.
        """
        ...

    def get_descriptive_stats(self) -> DescriptiveDatasetStats:
        texts = self.get_documents()
        document_lengths = np.array([len(text) for text in texts])

        mean = float(np.mean(document_lengths))
        std = float(np.std(document_lengths))
        return DescriptiveDatasetStats(
            mean_document_length=mean,
            std_document_length=std,
            num_documents=len(document_lengths),
        )

    def name_to_path(self) -> str:
        """
        Convert a name to a path.
        """
        name = self.name.replace("/", "__").replace(" ", "_")
        return name
