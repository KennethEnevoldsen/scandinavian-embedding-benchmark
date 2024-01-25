import random
from typing import Any

import datasets
from mteb.abstasks import AbsTaskClustering


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
            "category": "p2p",  # and S2P
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
        splits = ["train", "test", "validation"]

        documents = []
        labels = []
        label_col = "article_category"

        for split in splits:
            ds_split = self.dataset[split]
            documents.extend(ds_split["headline"])
            labels.extend(ds_split[label_col])

            documents.extend(ds_split["summary"])
            labels.extend(ds_split[label_col])

            documents.extend(ds_split["article"])
            labels.extend(ds_split[label_col])

        pairs = list(zip(documents, labels))

        rng = random.Random(42)  # local only seed
        rng.shuffle(pairs)
        documents, labels = list(zip(*pairs))

        n_pairs = len(documents)  # 114k examples
        n_splits = 10  # chosen semi-arbitrarily based on existing clustering tasks in MTEB  -
        # we could also make sure that the summary, article and headlines are in the same bins?
        n_per_split = n_pairs // n_splits

        documents = [documents[i : i + n_per_split] for i in range(0, n_pairs, n_per_split)][:-1]
        labels = [labels[i : i + n_per_split] for i in range(0, n_pairs, n_per_split)][:-1]

        ds = datasets.Dataset.from_dict({"sentences": documents, "labels": labels})
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
            "eval_splits": ["test"],  # 18k examples (what size do we want to for this?)
            "eval_langs": ["nb"],  # maybe nn too?
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
            # just keeping it all as one cluster - Could imagine there is a reasonable size limit? How to choose?
            ds[split] = datasets.Dataset.from_dict({"sentences": [documents[:100]], "labels": [labels[:100]]})

        self.dataset = datasets.DatasetDict(ds)

    @staticmethod
    def normalize_labels(labels: list[str]) -> list[str]:
        # Agreed on and debated in: https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/issues/83
        return [label.split(",")[0] for label in labels]