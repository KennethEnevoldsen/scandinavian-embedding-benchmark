from typing import Any

import datasets
from mteb.abstasks import AbsTaskRetrieval


class DanFever(AbsTaskRetrieval):
    @property
    def description(self) -> dict[str, Any]:
        return {
            "name": "DanFEVER",
            "hf_hub_name": "strombergnlp/danfever",
            "description": "A Danish dataset intended for misinformation research. It follows the same format as the English FEVER dataset.",
            "reference": "https://aclanthology.org/2021.nodalida-main.47/",
            "type": "Retrieval",
            "category": "p2p",
            "eval_splits": ["train"],
            "eval_langs": ["da"],
            "main_score": "ndcg_at_10",
            "revision": "5d01e3f6a661d48e127ab5d7e3aaa0dc8331438a",
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

        self.corpus = Dict[doc_id, Dict[str, str]] #id => dict with document data like title and text
        self.queries = Dict[query_id, str] #id => query
        self.relevant_docs = Dict[query_id, Dict[[doc_id, score]]
        """
        self.corpus = {}
        self.relevant_docs = {}
        self.queries = {}
        text2id = {}

        for split in self.dataset:
            self.corpus[split] = {}
            self.relevant_docs[split] = {}
            self.queries[split] = {}

            ds = self.dataset[split]
            claims = ds["claim"]
            evidences = ds["evidence_extract"]
            labels = ds["label"]
            class_labels = ds.features["label"].names

            for claim, evidence, label_id in zip(claims, evidences, labels):
                claim_is_supported = class_labels[label_id] == "Supported"

                sim = 1 if claim_is_supported else 0  # negative for refutes claims - is that what we want?

                if claim not in text2id:
                    text2id[claim] = str(len(text2id))
                if evidence not in text2id:
                    text2id[evidence] = len(text2id)

                claim_id = str(text2id[claim])
                evidence_id = str(text2id[evidence])

                self.queries[split][claim_id] = claim
                self.corpus[split][evidence_id] = {"title": "", "text": evidence}

                if claim_id not in self.relevant_docs[split]:
                    self.relevant_docs[split][claim_id] = {}

                self.relevant_docs[split][claim_id][evidence_id] = sim
