from typing import Any

import datasets
from mteb.abstasks import AbsTaskRetrieval


class SweFaqRetrieval(AbsTaskRetrieval):
    @property
    def description(self) -> dict[str, Any]:
        return {
            "name": "swefaq",
            "hf_hub_name": "AI-Sweden/SuperLim",
            "description": "A Swedish QA dataset derived from FAQ",
            "reference": "https://spraakbanken.gu.se/en/resources/superlim",
            "type": "Retrieval",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["sv"],
            "main_score": "ndcg_at_10",
            "revision": "7ebf0b4caa7b2ae39698a889de782c09e6f5ee56",
        }

    def load_data(self, **kwargs: dict):  # noqa: ARG002
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return

        self.dataset: datasets.DatasetDict = datasets.load_dataset(
            self.description["hf_hub_name"],
            "swefaq",  # chose the relevant subset
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
            self.queries[split] = {}
            self.relevant_docs[split] = {}
            self.corpus[split] = {}

            questions = ds["question"]
            ca_answers = ds["candidate_answer"]
            co_answers = ds["correct_answer"]

            n = 0
            for q, ca, co in zip(questions, ca_answers, co_answers):
                self.queries[split][str(n)] = q
                q_n = n
                n += 1
                if ca not in text2id:
                    text2id[ca] = n
                    self.corpus[split][str(n)] = {"title": "", "text": ca}
                    n += 1
                if co not in text2id:
                    text2id[co] = n
                    self.corpus[split][str(n)] = {"title": "", "text": co}
                    n += 1
                cor_n = text2id[co]

                self.relevant_docs[split][str(q_n)] = {
                    str(cor_n): 1,
                }  # only one correct match
