"""
Experiments:

Using two sample models (text-embedding-3-small, all-MiniLM-L6-v2) we get the following results:

Using 2 levels of labels (17 unique labels):
41.98, 23.71
Using 3 levels of labels (121 unique labels):
67.04, 53.39
Using 3 levels of labels (121 unique labels):
67.04, 53.39

As level 3 gives better results while also having more labels as well as more humanly well-defined labels we will use 3 levels.

Now changing for 4 batches of size 512 to 2 batches of size 1024:
Using 3 levels:
67.04, 53.39

As it does not change much we will use 4 batches of size 512.
"""

import random
from collections.abc import Iterable
from itertools import islice
from typing import Any, TypeVar

import datasets
from mteb.abstasks import AbsTaskClustering

T = TypeVar("T")


def batched(iterable: Iterable[T], n: int) -> Iterable[tuple[T, ...]]:
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


class SNLClustering(AbsTaskClustering):
    @property
    def description(self) -> dict[str, Any]:
        return {
            "name": "SNLClustering",
            "hf_hub_name": "navjordj/SNL_summarization",
            "description": "Webscrabed articles from the Norwegian lexicon 'Det Store Norske Leksikon'. Uses articles categories as clusters.",
            "reference": "https://huggingface.co/datasets/navjordj/SNL_summarization",
            "type": "Clustering",
            "category": "p2p",
            "eval_splits": ["test"],
            "eval_langs": ["nb"],
            "main_score": "v_measure",
            "revision": "3d3d27aa7af8941408cefc3991ada5d12a4273d1",
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
        label_col = "category"

        ds = {}
        for split in splits:
            ds_split = self.dataset[split]

            _label = self.normalize_labels(ds_split[label_col])
            documents.extend(ds_split["ingress"])
            labels.extend(_label)

            documents.extend(ds_split["article"])
            labels.extend(_label)

            assert len(documents) == len(labels)

            rng = random.Random(42)  # local only seed
            pairs = list(zip(documents, labels))
            rng.shuffle(pairs)
            documents, labels = (list(collection) for collection in zip(*pairs))

            # reduce size of dataset to not have too large datasets in the clustering task
            documents_batched = list(batched(documents, 512))[:4]
            labels_batched = list(batched(labels, 512))[:4]

            ds[split] = datasets.Dataset.from_dict(
                {
                    "sentences": documents_batched,
                    "labels": labels_batched,
                }
            )

        self.dataset = datasets.DatasetDict(ds)

    @staticmethod
    def normalize_labels(labels: list[str]) -> list[str]:
        # example label:
        # Store norske leksikon,Kunst og estetikk,Musikk,Klassisk musikk,Internasjonale dirigenter
        # When using 2 levels there is 17 unique labels
        # When using 3 levels there is 121 unique labels
        return [",".join(tuple(label.split(",")[:3])) for label in labels]
