from datetime import datetime
from typing import Any

import numpy as np
from datasets import DatasetDict, concatenate_datasets
from mteb import AbsTask
from mteb import __version__ as mteb_version

from ..result_dataclasses import TaskResult
from ..types import ArrayLike
from .model import Encoder
from .task import DescriptiveDatasetStats, Task


class MTEBTaskModel(Encoder):
    def __init__(self, mteb_model: Encoder, task: Task) -> None:
        self.mteb_model = mteb_model
        self.task = task

    def encode(self, texts: list[str], **kwargs: Any) -> ArrayLike:
        return self.mteb_model.encode(texts, task=self.task, **kwargs)


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
        self.task_type = mteb_desc["type"]
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

    def get_descriptive_stats(self) -> DescriptiveDatasetStats:
        ds: DatasetDict = self.load_data()
        texts = []
        for split in ds:
            for text_column in self._text_columns:
                texts += ds[split][text_column]

        document_lengths = np.array([len(text) for text in texts])

        mean = float(np.mean(document_lengths))
        std = float(np.std(document_lengths))
        return DescriptiveDatasetStats(
            mean_document_length=mean,
            std_document_length=std,
            num_documents=len(document_lengths),
        )

    def evaluate(self, model: Encoder) -> TaskResult:
        split = self.mteb_task.description["eval_splits"][0]
        task_model = MTEBTaskModel(model, self)
        scores = self.mteb_task.evaluate(task_model, split=split)
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
