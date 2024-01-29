from typing import Any

import datasets
from mteb.abstasks import AbsTaskRetrieval


class TV2Nordretrieval(AbsTaskRetrieval):
    @property
    def description(self) -> dict[str, Any]:
        return {
            "name": "TV2Nordretrieval",
            "hf_hub_name": "alexandrainst/nordjylland-news-summarization",
            "description": "News Article and corresponding summaries extracted from the Danish newspaper TV2 Nord.",
            "reference": "https://huggingface.co/datasets/alexandrainst/nordjylland-news-summarization",
            "type": "Retrieval",
            "category": "p2p",
            "eval_splits": ["test"],
            "eval_langs": ["da"],
            "main_score": "ndcg_at_10",
            "revision": "80cdb115ec2ef46d4e926b252f2b59af62d6c070",
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

    def dataset_transform(self) -> None:
        """
        and transform to a retrieval datset, which have the following attributes

        self.corpus = Dict[doc_id, Dict[str, str]] #id => dict with document datas like title and text
        self.queries = Dict[query_id, str] #id => query
        self.relevant_docs = Dict[query_id, Dict[[doc_id, score]]
        """
        self.corpus = {}
        self.relevant_docs = {}
        self.queries = {}
        text2id = {}

        for split in self.dataset:
            ds: datasets.Dataset = self.dataset[split]  # type: ignore
            ds = ds.shuffle(seed=42)
            ds = ds.select(range(2048))  # limit the dataset size to make sure the task does not take too long to run
            self.queries[split] = {}
            self.relevant_docs[split] = {}
            self.corpus[split] = {}

            summary = ds["summary"]
            article = ds["text"]

            n = 0
            for summ, art in zip(summary, article):
                self.queries[split][str(n)] = summ
                q_n = n
                n += 1
                if art not in text2id:
                    text2id[art] = n
                    self.corpus[split][str(n)] = {"title": "", "text": art}
                    n += 1
                cor_n = text2id[art]

                self.relevant_docs[split][str(q_n)] = {str(text2id[art]): 1}  # only one correct matches
