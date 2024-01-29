import random
from typing import Any, TypeVar

import datasets
from mteb.abstasks import AbsTaskSTS
from seb.interfaces.mteb_task import MTEBTask
from seb.interfaces.task import Task
from seb.registries import tasks

T = TypeVar("T")


class SwednSummarizationSTS(AbsTaskSTS):
    def load_data(self, **kwargs: dict):  # noqa: ARG002
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return

        self.dataset: datasets.DatasetDict = datasets.load_dataset(
            self.description["hf_hub_name"],
            "swedn",  # chose the relevant subset
            revision=self.description.get("revision"),
        )  # type: ignore

        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self) -> None:
        self.dataset = self.dataset.rename_column("summary", "sentence2")
        self.dataset = self.dataset.rename_column("article", "sentence1")
        self.dataset = self.dataset.remove_columns(["id", "headline", "article_category"])
        self.dataset = self.dataset.shuffle(seed=42)

        # add score column
        for split in self.dataset:
            ds_split = self.dataset[split]
            ds_split = ds_split.add_column("score", [1] * len(ds_split))  # type: ignore
            self.dataset[split] = ds_split

            # Add a wrongly mapped examples. To ensure tasks in non-trivial
            summaries = ds_split["sentence2"]
            articles = ds_split["sentence1"]
            scores = ds_split["score"]
            mismatched_summaries = self.sattolo_cycle(summaries)

            # add all the mismatched examples as negative examples
            mismatched_ds = datasets.Dataset.from_dict(
                {
                    "sentence1": articles,
                    "sentence2": mismatched_summaries,
                    "score": ([0] * len(articles)),
                }
            )
            mismatched_ds = mismatched_ds.shuffle(seed=42)
            self.dataset[split] = datasets.concatenate_datasets([ds_split.select(range(1024)), mismatched_ds.select(range(1024))])

    @property
    def description(self) -> dict[str, Any]:
        return {
            "name": "Swedn",
            "hf_hub_name": "sbx/superlim-2",
            "description": "News Article Summary Semantic Similarity Estimation.",
            "reference": "https://spraakbanken.gu.se/en/resources/swedn",
            "type": "STS",
            "category": "p2p",
            "eval_splits": ["test"],
            "eval_langs": ["sv"],
            "main_score": "cosine_spearman",
            "min_score": 0,
            "max_score": 1,
            "revision": "ef1661775d746e0844b299164773db733bdc0bf6",
        }

    @staticmethod
    def sattolo_cycle(items: list[T]) -> list[T]:
        """
        The Sattolo cycle is a simple algorithm for randomly shuffling an array in-place.
        It ensures that the element i, will not be in the ith position of the result.
        """
        rng = random.Random(42)
        for i in range(len(items) - 1, 0, -1):
            j = rng.randint(0, i - 1)
            items[i], items[j] = items[j], items[i]
        return items


@tasks.register("SwednSTS")
def create_swedn_sts() -> Task:
    task = MTEBTask(SwednSummarizationSTS())
    task.name = "SwednSTS"
    task.version = "0.0.1"
    task.domain = ["non-fiction", "news"]
    return task
