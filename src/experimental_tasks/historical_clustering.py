import random
from collections.abc import Iterable
from itertools import islice
from typing import Any, TypeVar

import datasets
from mteb.abstasks import AbsTaskClustering

from seb.interfaces.mteb_task import MTEBTask
from seb.interfaces.task import Task
from seb.registries import tasks

T = TypeVar("T")


def batched(iterable: Iterable[T], n: int) -> Iterable[tuple[T, ...]]:
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


class HistoricalDanishClustering(AbsTaskClustering):
    @property
    def description(self) -> dict[str, Any]:
        return {
            "name": "HistoricalDanishClustering",
            "hf_hub_name": "kardosdrur/historical-danish-clustering",
            "description": "Subset of the MeMo corpus for clustering historical texts.",
            "reference": "https://huggingface.co/datasets/MiMe-MeMo/Corpus-v1.1",
            "type": "Clustering",
            "category": "p2p",
            "eval_splits": ["train"],
            "eval_langs": ["da"],
            "main_score": "v_measure",
            "revision": "3312e730dfa51f9e411dd4f178e242d00063b363",
        }

    def load_data(self, **kwargs: dict):  # noqa: ARG002
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return

        self.dataset: datasets.DatasetDict = datasets.load_dataset(
            self.description["hf_hub_name"],
            revision=self.description.get("revision"),
        )  # type: ignore

        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self):
        splits = self.description["eval_splits"]

        documents: list = []
        labels: list = []
        label_col = "full_title"

        ds = {}
        for split in splits:
            ds_split = self.dataset[split]

            _label = ds_split[label_col]
            documents.extend(ds_split["text"])
            labels.extend(_label)

            assert len(documents) == len(labels)

            rng = random.Random(42)  # local only seed
            pairs = list(zip(documents, labels))
            rng.shuffle(pairs)
            documents, labels = (list(collection) for collection in zip(*pairs))
            ds[split] = datasets.Dataset.from_dict(
                {
                    "sentences": [documents],
                    "labels": [labels],
                }
            )
        self.dataset = datasets.DatasetDict(ds)


@tasks.register("HistoricalDanishClustering")
def create_historical() -> Task:
    task = MTEBTask(HistoricalDanishClustering())
    task.name = "HistoricalDanishClustering"
    task.domain = ["poetry", "fiction"]
    task.task_subtypes = ["Thematic Clustering"]
    return task
