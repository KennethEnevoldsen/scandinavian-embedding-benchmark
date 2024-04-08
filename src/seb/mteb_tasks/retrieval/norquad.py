"""
The dataset is quite similar to the SNL dataset (both are wikistyle), however NorQuad actually uses questions, while the other is just headline,
article pairs.
"""

from typing import Any

import datasets
from mteb.abstasks import AbsTaskRetrieval


class NorQuadRetrieval(AbsTaskRetrieval):
    @property
    def description(self) -> dict[str, Any]:
        return {
            "name": "NorQuadRetrieval",
            "hf_hub_name": "mteb/norquad_retrieval",
            "description": "Human-created question for Norwegian wikipedia passages.",
            "reference": "https://aclanthology.org/2023.nodalida-1.17/",
            "type": "Retrieval",
            "category": "p2p",
            "eval_splits": ["test"],
            "eval_langs": ["nb"],
            "main_score": "ndcg_at_10",
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
            max_samples = min(1024, len(ds))
            ds = ds.select(range(max_samples))  # limit the dataset size to make sure the task does not take too long to run
            self.queries[split] = {}
            self.relevant_docs[split] = {}
            self.corpus[split] = {}

            question = ds["question"]
            context = ds["context"]
            answer = [a["text"][0] for a in ds["answers"]]

            n = 0
            for q, cont, ans in zip(question, context, answer):
                self.queries[split][str(n)] = q
                q_n = n
                n += 1
                if cont not in text2id:
                    text2id[cont] = n
                    self.corpus[split][str(n)] = {"title": "", "text": cont}
                    n += 1
                if ans not in text2id:
                    text2id[ans] = n
                    self.corpus[split][str(n)] = {"title": "", "text": ans}
                    n += 1

                self.relevant_docs[split][str(q_n)] = {str(text2id[ans]): 1, str(text2id[cont]): 1}  # only two correct matches
