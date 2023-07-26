from datetime import datetime
from typing import List, Protocol

from mteb import AbsTask

from .model_interface import ModelInterface
from .results import TaskResult


class Task(Protocol):
    name: str
    main_score: str
    description: str
    reference: str
    version: str
    languages: List[str]

    def evaluate(self, model: ModelInterface) -> TaskResult:
        """
        Evaluates a Sentence Embedding Model on the task.

        Args:
            model: A sentence embedding model.

        Returns:
            A TaskResult object.
        """
        ...


class MTEBTask(Task):
    def __init__(self, mteb_task: AbsTask) -> None:
        self.mteb_task = mteb_task
        mteb_desc = self.mteb_task.description
        self.main_score = mteb_desc.main_score
        self.name = mteb_desc.name
        self.description = mteb_desc.description
        self.version = f"{mteb_desc.dataset_revision}_{mteb_desc.mteb_version}"
        self.reference = mteb_desc.reference
        self.languages = mteb_desc.languages

    def evaluate(self, model: ModelInterface) -> TaskResult:
        scores = self.mteb_task.evaluate(model)
        if scores is None:
            raise ValueError("MTEBTask evaluation failed.")
        split = self.mteb_task.description.split[0]

        time_of_run: datetime = datetime.now()

        task_result = TaskResult(
            task_name=self.name,
            task_description=self.description,
            task_version=self.version,
            time_of_run=time_of_run,
            scores=scores[split],
            main_score=self.main_score,
        )

        return task_result
