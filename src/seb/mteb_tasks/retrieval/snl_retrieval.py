"""
Experiments:

Using two sample models (text-embedding-3-small, all-MiniLM-L6-v2) we get the following results:

96.07, 64.67

if we then remove the ingress from the corpus we get:
92.71, 51.61
The reason why we might want to remove the ingress is that it almost always start with headline.

As the scores are indeed slightly lower we will ignore the ingress as the task becomes too easy.
"""

from typing import Any

import datasets
from mteb.abstasks import AbsTaskRetrieval


class SNLRetrieval(AbsTaskRetrieval):
    @property
    def description(self) -> dict[str, Any]:
        return {
            "name": "SNLClustering",
            "hf_hub_name": "navjordj/SNL_summarization",
            "description": "Webscrabed articles and ingresses from the Norwegian lexicon 'Det Store Norske Leksikon'.",
            "reference": "https://huggingface.co/datasets/navjordj/SNL_summarization",
            "type": "Retrieval",
            "category": "p2p",
            "eval_splits": ["test"],
            "eval_langs": ["nb"],
            "main_score": "ndcg_at_10",
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

            self.queries[split] = {}
            self.relevant_docs[split] = {}
            self.corpus[split] = {}

            headline = ds["headline"]
            article = ds["article"]

            n = 0
            for headl, art in zip(headline, article):
                self.queries[split][str(n)] = headl
                q_n = n
                n += 1
                if art not in text2id:
                    text2id[art] = n
                    self.corpus[split][str(n)] = {"title": "", "text": art}
                    n += 1
                self.relevant_docs[split][str(q_n)] = {str(text2id[art]): 1}  # only one correct matches
