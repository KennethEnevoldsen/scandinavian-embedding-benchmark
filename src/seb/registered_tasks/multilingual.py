from datetime import datetime
from functools import partial
from typing import Any

import numpy as np
from datasets import DatasetDict, concatenate_datasets

from seb.interfaces.model import Encoder
from seb.interfaces.mteb_task import MTEBTask
from seb.interfaces.task import DescriptiveDatasetStats, Task
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
    task.task_subtypes = ["Dialog Systems", "Intent Classification"]
    return task


@tasks.register("Massive Scenario")
def create_massive_scenario() -> Task:
    from mteb import MassiveScenarioClassification

    task = MTEBTask(MassiveScenarioClassification())
    task.name = "Massive Scenario"
    task.languages = ["da", "nb", "sv"]
    task.mteb_task.langs = ["da", "nb", "sv"]  # type: ignore
    task.domain = ["spoken"]
    task.task_subtypes = ["Dialog Systems", "Scenario Classification"]
    return task


@tasks.register("ScaLA")
def create_scala() -> Task:
    from mteb import ScalaDaClassification, ScalaNbClassification, ScalaNnClassification, ScalaSvClassification, __version__

    # this is a temporary fix until the mteb library is updated
    class ScalaDaClassificationCustom(ScalaDaClassification):
        @property
        def description(self) -> dict[str, Any]:
            return {
                "name": "ScalaDaClassification",
                "hf_hub_name": "mteb/scala_da_classification",
                "description": "A modified version of DDT modified for linguistic acceptability classification",
                "reference": "https://aclanthology.org/2023.nodalida-1.20/",
                "type": "Classification",
                "category": "s2s",
                "eval_splits": ["test"],
                "eval_langs": ["da"],
                "main_score": "accuracy",
                "n_experiments": 10,
                "samples_per_label": 16,
            }

    class ScalaNbClassificationCustom(ScalaNbClassification):
        @property
        def description(self) -> dict[str, Any]:
            return {
                "name": "ScalaNbClassification",
                "hf_hub_name": "mteb/scala_nb_classification",
                "description": "A Norwegian dataset for linguistic acceptability classification for Bokmål",
                "reference": "https://aclanthology.org/2023.nodalida-1.20/",
                "type": "Classification",
                "category": "s2s",
                "eval_splits": ["test"],
                "eval_langs": ["no", "nb"],
                "main_score": "accuracy",
                "n_experiments": 10,
                "samples_per_label": 16,
            }

    class ScalaSvClassificationCustom(ScalaSvClassification):
        @property
        def description(self) -> dict[str, Any]:
            return {
                "name": "ScalaSvClassification",
                "hf_hub_name": "mteb/scala_sv_classification",
                "description": "A Swedish dataset for linguistic acceptability classification",
                "reference": "https://aclanthology.org/2023.nodalida-1.20/",
                "type": "Classification",
                "category": "s2s",
                "eval_splits": ["test"],
                "eval_langs": ["sv"],
                "main_score": "accuracy",
                "n_experiments": 10,
                "samples_per_label": 16,
            }

    class ScalaNnClassificationCustom(ScalaNnClassification):
        @property
        def description(self) -> dict[str, Any]:
            return {
                "name": "ScalaNnClassification",
                "hf_hub_name": "mteb/scala_nn_classification",
                "description": "A Norwegian dataset for linguistic acceptability classification for Nynorsk",
                "reference": "https://aclanthology.org/2023.nodalida-1.20/",
                "type": "Classification",
                "category": "s2s",
                "eval_splits": ["test"],
                "eval_langs": ["no", "nn"],
                "main_score": "accuracy",
                "n_experiments": 10,
                "samples_per_label": 16,
            }

    class ScalaTask(Task):
        def __init__(self) -> None:
            self.mteb_tasks = {
                "da": ScalaDaClassificationCustom(),
                "nb": ScalaNbClassificationCustom(),
                "sv": ScalaSvClassificationCustom(),
                "nn": ScalaNnClassificationCustom(),
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
            self.task_subtypes = ["Linguistic Acceptability"]

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

        def get_documents(self) -> list[str]:
            ds = self.load_data()
            texts = []
            splits = self.get_splits()
            assert len(splits) >= 1, "No splits found in MTEB task."

            for split in splits:
                for text_column in self._text_columns:
                    texts += ds[split][text_column]
            return texts

        def get_descriptive_stats(self) -> DescriptiveDatasetStats:
            texts = self.get_documents()
            document_lengths = np.array([len(text) for text in texts])

            mean = np.mean(document_lengths)
            std = np.std(document_lengths)
            return DescriptiveDatasetStats(
                mean_document_length=float(mean),
                std_document_length=float(std),
                num_documents=len(document_lengths),
            )

        def evaluate(self, model: Encoder) -> TaskResult:
            scores = {}
            # Infusing task into encode()
            original_encode = model.encode
            try:
                model.encode = partial(model.encode, task=self)
                for lang, mteb_task in self.mteb_tasks.items():
                    mteb_task.load_data()
                    score = mteb_task.evaluate(model)
                    scores[lang] = score
                model.encode = original_encode
            except Exception as e:
                raise e
            finally:
                # Resetting encode to original
                model.encode = original_encode

            return TaskResult(
                task_name=self.name,
                task_version=self.version,
                time_of_run=datetime.now(),
                scores=scores,
                task_description=self.description,
                main_score=self.main_score,
            )

        def get_splits(self) -> list[str]:
            splits = []
            for lang, mteb_task in self.mteb_tasks.items():  # noqa: B007
                splits += mteb_task.description["eval_splits"]
            return list(set(splits))

    return ScalaTask()


@tasks.register("Language Identification")
def create_language_identification() -> Task:
    from mteb import NordicLangClassification

    task = MTEBTask(NordicLangClassification())
    task.name = "Language Identification"
    task.domain = ["wiki"]
    task.task_subtypes = ["Language Identification"]

    return task
