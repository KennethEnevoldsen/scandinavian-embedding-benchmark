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


class SwednClustering(AbsTaskClustering):
    @property
    def description(self) -> dict[str, Any]:
        return {
            "name": "SwednClustering",
            "hf_hub_name": "sbx/superlim-2",
            "description": "The SWE-DN corpus is based on 1,963,576 news articles from the Swedish newspaper "
            + "Dagens Nyheter (DN) during the years 2000--2020. The articles are filtered to resemble the CNN/DailyMail"
            + " dataset both regarding textual structure. This dataset uses the category labels as clusters.",
            "reference": "https://spraakbanken.gu.se/en/resources/swedn",
            "type": "Clustering",
            "category": "p2p",
            "eval_splits": ["all"],
            "eval_langs": ["sv"],
            "main_score": "v_measure",
            "revision": "ef1661775d746e0844b299164773db733bdc0bf6",
        }

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

    def dataset_transform(self):
        """
        The article_category clusters differ between the splits (with the test set only having 1 cluster). Therefore we combine it all into one
        cluster.
        """
        splits = ["train", "validation"]
        # performance of sample models with test set: 8.74, 2.43 -removing test-> 11.26, 4.27
        # this is due to the test set only having 1 cluster which is "other"

        headlines = []
        summaries = []
        articles = []
        labels = []
        label_col = "article_category"

        for split in splits:
            ds_split = self.dataset[split]
            headlines.extend(ds_split["headline"])
            labels.extend(ds_split[label_col])

            summaries.extend(ds_split["summary"])
            labels.extend(ds_split[label_col])

            articles.extend(ds_split["article"])
            labels.extend(ds_split[label_col])

        rng = random.Random(42)  # local only seed

        clusters_text = []
        clusters_labels = []
        doc_types = [summaries, articles]
        # Note that headlines is excluded:
        # Scores of sample models with headlines: 11.26, 4.27 -removing headlines-> 16.43, 4.31
        # as headlines are soo short it is hard to meaningfully cluster them even for humans.
        for text in doc_types:
            pairs = list(zip(text, labels))
            rng.shuffle(pairs)
            # reduce size of dataset to not have too large datasets in the clustering task
            pairs_batched = list(batched(pairs, 512))
            texts1, labels2 = list(zip(*pairs_batched[0]))
            texts2, labels2 = list(zip(*pairs_batched[1]))

            clusters_text.extend([texts1, texts2])
            clusters_labels.extend([labels2, labels2])
        ds = datasets.Dataset.from_dict({"sentences": clusters_text, "labels": clusters_labels})
        self.dataset = datasets.DatasetDict({"all": ds})


class VGSummarizationClustering(AbsTaskClustering):
    @property
    def description(self) -> dict[str, Any]:
        return {
            "name": "VGSummarizationClustering",
            "hf_hub_name": "navjordj/VG_summarization",
            "description": "Articles and ingresses from VG news articles extracted from Norsk Aviskorpus. Uses articles classes as clusters.",
            "reference": "https://huggingface.co/datasets/navjordj/VG_summarization",
            "type": "Clustering",
            "category": "p2p",
            "eval_splits": ["test"],
            "eval_langs": ["nb"],
            "main_score": "v_measure",
            "revision": "d4c5a8ba10ae71224752c727094ac4c46947fa29",
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

        documents = []
        labels = []
        label_col = "classes"

        ds = {}
        for split in splits:
            ds_split = self.dataset[split]

            _label = self.normalize_labels(ds_split[label_col])
            documents.extend(ds_split["title"])
            labels.extend(_label)

            documents.extend(ds_split["ingress"])
            labels.extend(_label)

            documents.extend(ds_split["article"])
            labels.extend(_label)

            assert len(documents) == len(labels)

            rng = random.Random(1111)  # local only seed
            # resampling changes scores from 12.68, 11.30, 12.65 (sample model)
            pairs = list(zip(documents, labels))
            rng.shuffle(pairs)
            documents, labels = list(zip(*pairs))

            # reduce size of dataset to not have too large datasets in the clustering task
            documents_batched = list(batched(documents, 512))[:4]
            labels_batched = list(batched(labels, 512))[:4]
            # See:
            # https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/pull/96
            # for a discussion on sizes

            ds[split] = datasets.Dataset.from_dict(
                {
                    "sentences": documents_batched,
                    "labels": labels_batched,
                }
            )

        self.dataset = datasets.DatasetDict(ds)

    @staticmethod
    def normalize_labels(labels: list[str]) -> list[str]:
        # Agreed on and debated in: https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/issues/83
        return [label.split(",")[0] for label in labels]
