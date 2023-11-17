from datetime import datetime
from typing import Any, Protocol, runtime_checkable

import numpy as np
from datasets import DatasetDict, concatenate_datasets
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
        domain: The domains of the task. Should be one of the categories listed on https://universaldependencies.org

    """

    name: str
    main_score: str
    description: str
    reference: str
    version: str
    languages: list[str]
    domain: list[str]

    def evaluate(self, model: ModelInterface) -> TaskResult:
        """
        Evaluates a Sentence Embedding Model on the task.

        Args:
            model: A sentence embedding model.

        Returns:
            A TaskResult object.
        """
        ...

    def get_descriptive_stats(self) -> dict[str, Any]:
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
        self.type = mteb_desc["type"]
        self.domain = []
        self._text_columns = ["text"]

    def load_data(self) -> DatasetDict:
        self.mteb_task.load_data()
        ds = {}
        # check if it is split into dataset[lang][split] or to dataset[split]
        keys = list(self.mteb_task.dataset.keys())  # type: ignore
        contains_lang = isinstance(self.mteb_task.dataset[keys[0]], DatasetDict)  # type: ignore

        if not contains_lang:  # it is split into dataset[split]
            for split in self.mteb_task.description["eval_splits"]:
                ds[split] = self.mteb_task.dataset[split]  # type: ignore
        else:  # it is split into dataset[lang][split] and we convert to dataset[split]
            for lang in self.languages:
                for split in self.mteb_task.description["eval_splits"]:
                    if split not in ds:
                        ds[split] = self.mteb_task.dataset[lang][split]  # type: ignore
                    else:
                        ds[split] = concatenate_datasets(
                            [ds[split], self.mteb_task.dataset[lang][split]],  # type: ignore
                        )

        return DatasetDict(ds)

    def get_descriptive_stats(self) -> dict[str, Any]:
        ds = self.load_data()
        texts = []
        for split in ds:
            for text_column in self._text_columns:
                texts += ds[split][text_column]

        document_lengths = [len(text) for text in texts]

        mean = np.mean(document_lengths)
        std = np.std(document_lengths)
        return {
            "mean_document_length": mean,
            "std_document_length": std,
            "num_documents": len(document_lengths),
        }

    def evaluate(self, model: ModelInterface) -> TaskResult:
        split = self.mteb_task.description["eval_splits"][0]
        scores = self.mteb_task.evaluate(model, split=split)
        if scores is None:
            raise ValueError("MTEBTask evaluation failed.")

        # there is only one split in all MTEB tasks in SEB

        time_of_run: datetime = datetime.now()

        scores = scores.get(split, scores)
        score_is_nested = isinstance(scores[next(iter(scores.keys()))], dict)
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
