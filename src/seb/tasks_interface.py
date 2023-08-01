from datetime import datetime
from typing import List, Protocol, runtime_checkable

from mteb import AbsTask
from mteb import __version__ as mteb_version

from .model_interface import ModelInterface
from .result_dataclasses import TaskResult


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
        
    """
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
        self.main_score = mteb_desc["main_score"]
        self.name = mteb_desc["name"]
        self.description = mteb_desc["description"]
        self.version = f"{mteb_version}"
        self.reference = mteb_desc["reference"]
        self.languages = mteb_desc["eval_langs"]

    def evaluate(self, model: ModelInterface) -> TaskResult:
        split = self.mteb_task.description["eval_splits"][0]
        scores = self.mteb_task.evaluate(model, split=split)
        if scores is None:
            raise ValueError("MTEBTask evaluation failed.")

        # there is only one split in all MTEB tasks in SEB

        time_of_run: datetime = datetime.now()

        scores = scores.get(split, scores)
        score_is_nested = isinstance(scores[list(scores.keys())[0]], dict)
        if not score_is_nested:
            _scores = {lang: scores for lang in self.languages}
            scores = _scores

        task_result = TaskResult(
            task_name=self.name,
            task_description=self.description,
            task_version=self.version,
            time_of_run=time_of_run,
            scores=scores,
            main_score=self.main_score,
        )

        return task_result
