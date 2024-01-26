import random
from typing import Any, TypeVar

import datasets
from mteb.abstasks import AbsTaskBitextMining, AbsTaskClassification, AbsTaskRetrieval, AbsTaskSTS

T = TypeVar("T")


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


class NorwegianParliamentClassification(AbsTaskClassification):
    # this changes the description of the tasks but otherwise is the same as the task in the MTEB benchmark
    # once we have collected a few MTEB tasks not in the MTEB benchmark we can add them back to the benchmark.
    @property
    def description(self) -> dict[str, Any]:
        return {
            "name": "NorwegianParliament",
            "hf_hub_name": "NbAiLab/norwegian_parliament",
            "description": "Norwegian parliament speeches annotated with the party of the speaker (`Sosialistisk Venstreparti` vs `Fremskrittspartiet`)",
            "reference": "https://huggingface.co/datasets/NbAiLab/norwegian_parliament",
            "type": "Classification",
            "category": "s2s",
            "eval_splits": ["test", "validation"],
            "eval_langs": ["nb"],  # assumed to be bokmål
            "main_score": "accuracy",
            "n_experiments": 10,
            "samples_per_label": 16,
            "revision": "f7393532774c66312378d30b197610b43d751972",
        }


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


class SwednRetrieval(AbsTaskRetrieval):
    @property
    def description(self) -> dict[str, Any]:
        return {
            "name": "Swedn",
            "hf_hub_name": "sbx/superlim-2",
            "description": "News Article Summary Semantic Similarity Estimation.",
            "reference": "https://spraakbanken.gu.se/en/resources/swedn",
            "type": "Retrieval",
            "category": "p2p",
            "eval_splits": ["test"],
            "eval_langs": ["sv"],
            "main_score": "ndcg_at_10",
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
            ds = ds.select(range(1024))  # limit the dataset size to make sure the task does not take too long to run
            self.queries[split] = {}
            self.relevant_docs[split] = {}
            self.corpus[split] = {}

            headline = ds["headline"]
            summary = ds["summary"]
            article = ds["article"]

            n = 0
            for headl, summ, art in zip(headline, summary, article):
                self.queries[split][str(n)] = headl
                q_n = n
                n += 1
                if summ not in text2id:
                    text2id[summ] = n
                    self.corpus[split][str(n)] = {"title": "", "text": summ}
                    n += 1
                if art not in text2id:
                    text2id[art] = n
                    self.corpus[split][str(n)] = {"title": "", "text": art}
                    n += 1
                cor_n = text2id[art]

                self.relevant_docs[split][str(q_n)] = {str(text2id[art]): 1, str(text2id[summ]): 1}  # only two correct matches


class NorwegianCourtsBitextMining(AbsTaskBitextMining):
    @property
    def description(self) -> dict[str, Any]:
        return {
            "name": "NorwegianCourtsBitextMining",
            "hf_hub_name": "kardosdrur/norwegian-courts",
            "description": "Nynorsk and Bokmål parallel corpus from Norwegian courts. "
            + "Norway has two standardised written languages. "
            + "Bokmål is a variant closer to Danish, while Nynorsk was created to resemble "
            + "regional dialects of Norwegian.",
            "reference": "https://opus.nlpl.eu/ELRC-Courts_Norway-v1.php",
            "type": "BitextMining",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["nb", "nn"],
            "main_score": "f1",
            "revision": "3bc5cfb4ec514264fe2db5615fac9016f7251552",
        }

    def load_data(self, **kwargs: Any) -> None:  # noqa: ARG002
        """
        Load dataset from HuggingFace hub and convert it to the standard format.
        """
        if self.data_loaded:
            return

        self.dataset = datasets.load_dataset(
            self.description["hf_hub_name"],
            revision=self.description.get("revision", None),
        )
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self) -> None:
        # Convert to standard format
        self.dataset = self.dataset.rename_column("nb", "sentence1")
        self.dataset = self.dataset.rename_column("nn", "sentence2")
