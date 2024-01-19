from datetime import datetime
from typing import Any

import numpy as np
from datasets import DatasetDict, concatenate_datasets

from seb.interfaces.model import Encoder
from seb.interfaces.mteb_task import MTEBTask
from seb.interfaces.task import Task
from seb.registries import tasks
from seb.result_dataclasses import TaskResult


@tasks.register("Massive Intent")
def create_massive_intent() -> Task:
    from mteb import MassiveIntentClassification

    task = MTEBTask(MassiveIntentClassification())
    task.name = "Massive Intent"
    task.languages = ["da", "nb", "sv"]
    task.mteb_task.langs = ["da", "nb", "sv"]  # type: ignore
    task.domain = ["spoken"]
    return task


@tasks.register("Massive Scenario")
def create_massive_scenario() -> Task:
    from mteb import MassiveScenarioClassification

    task = MTEBTask(MassiveScenarioClassification())
    task.name = "Massive Scenario"
    task.languages = ["da", "nb", "sv"]
    task.mteb_task.langs = ["da", "nb", "sv"]  # type: ignore
    task.domain = ["spoken"]
    return task


@tasks.register("ScaLA")
def create_scala() -> Task:
    from mteb import (
        ScalaDaClassification,
        ScalaNbClassification,
        ScalaNnClassification,
        ScalaSvClassification,
        __version__,
    )

    class ScalaTask(Task):
        def __init__(self) -> None:
            self.mteb_tasks = {
                "da": ScalaDaClassification(),
                "nb": ScalaNbClassification(),
                "sv": ScalaSvClassification(),
                "nn": ScalaNnClassification(),
            }
            self.main_score = "accuracy"
            self.name = "ScaLA"
            self.description = "A linguistic acceptability task for Danish, Norwegian Bokmål Norwegian Nynorsk and Swedish."
            self.version = __version__
            self.reference = "https://aclanthology.org/2023.nodalida-1.20/"
            self.languages = ["da", "nb", "sv", "nn"]
            self.domain = ["fiction", "news", "non-fiction", "spoken", "blog"]
            self._text_columns = ["text"]
            self.task_type = "Classification"

        def load_data(self) -> DatasetDict:
            ds = {}
            for lang, mteb_task in self.mteb_tasks.items():  # noqa: B007
                mteb_task.load_data()
                for split in mteb_task.dataset:
                    if split not in ds:
                        ds[split] = mteb_task.dataset[split]
                    else:
                        ds[split] = concatenate_datasets(
                            [ds[split], mteb_task.dataset[split]],
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

        def evaluate(self, model: Encoder) -> TaskResult:
            scores = {}
            for lang, mteb_task in self.mteb_tasks.items():
                mteb_task.load_data()
                score = mteb_task.evaluate(model)
                scores[lang] = score

            return TaskResult(
                task_name=self.name,
                task_version=self.version,
                time_of_run=datetime.now(),
                scores=scores,
                task_description=self.description,
                main_score=self.main_score,
            )

    return ScalaTask()


@tasks.register("Language Identification")
def create_language_identification() -> Task:
    from mteb import NordicLangClassification

    task = MTEBTask(NordicLangClassification())
    task.name = "Language Identification"
    task.domain = ["wiki"]

    return task
