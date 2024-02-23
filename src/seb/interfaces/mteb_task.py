from datetime import datetime
from functools import partial
from typing import Any, Union

import numpy as np
from datasets import DatasetDict, concatenate_datasets
from mteb import AbsTask
from mteb import __version__ as mteb_version

from ..result_dataclasses import TaskResult
from .model import Encoder
from .task import DescriptiveDatasetStats, Task


class MTEBTask(Task):
    def __init__(self, mteb_task: AbsTask) -> None:
        self.mteb_task = mteb_task
        mteb_desc = self.mteb_task.description
        self.main_score = mteb_desc["main_score"]
        self.name = mteb_desc["name"]
        self.description = mteb_desc["description"]
        self.version = f"{mteb_version}"
        self.reference = mteb_desc["reference"]
        self.languages = mteb_desc["eval_langs"]
        self.task_type = mteb_desc["type"]
        self.domain = []
        self._text_columns = ["text"]
        self.task_subtypes = []

    def load_data(self) -> DatasetDict:
        self.mteb_task.load_data()
        ds = {}
        # check if it is split into dataset[lang][split] or to dataset[split]
        keys = list(self.mteb_task.dataset.keys())  # type: ignore
        contains_lang = isinstance(self.mteb_task.dataset[keys[0]], DatasetDict)  # type: ignore

        if not contains_lang:  # it is split into dataset[split]
            for split in self.mteb_task.description["eval_splits"]:
                ds[split] = self.mteb_task.dataset[split]  # type: ignore
        else:  # it is split into dataset[lang][split] and we convert to dataset[split]
            for lang in self.languages:
                for split in self.mteb_task.description["eval_splits"]:
                    if split not in ds:
                        ds[split] = self.mteb_task.dataset[lang][split]  # type: ignore
                    else:
                        ds[split] = concatenate_datasets(
                            [ds[split], self.mteb_task.dataset[lang][split]],  # type: ignore
                        )

        return DatasetDict(ds)

    def get_documents(self) -> list[str]:
        ds: DatasetDict = self.load_data()
        texts = []
        splits = self.mteb_task.description["eval_splits"]
        assert len(splits) >= 1, "No splits found in MTEB task."

        for split in splits:
            if self.task_type == "Retrieval":
                _corpus = self.mteb_task.corpus[split]  # type: ignore
                _queries = self.mteb_task.queries[split]  # type: ignore
                texts = [f"{text['title']} {text['text']}" for text in _corpus.values()]
                texts += list(_queries.values())
            elif self.task_type == "Clustering":
                for text_column in self._text_columns:
                    texts += [text for tl in ds[split][text_column] for text in tl]
            else:
                for text_column in self._text_columns:
                    texts += ds[split][text_column]

        return texts

    def get_descriptive_stats(self) -> DescriptiveDatasetStats:
        texts = self.get_documents()
        document_lengths = np.array([len(text) for text in texts])

        mean = float(np.mean(document_lengths))
        std = float(np.std(document_lengths))
        return DescriptiveDatasetStats(
            mean_document_length=mean,
            std_document_length=std,
            num_documents=len(document_lengths),
        )

    def format_scores(self, raw_scores: dict[str, Any], split: str) -> dict[str, Any]:
        if self.task_type == "STS":
            raw_scores = {
                "cosine_spearman": raw_scores["cos_sim"]["spearman"],
                "cosine_pearson": raw_scores["cos_sim"]["pearson"],
                "euclidean_spearman": raw_scores["euclidean"]["spearman"],
                "euclidean_pearson": raw_scores["euclidean"]["pearson"],
                "manhattan_spearman": raw_scores["manhattan"]["spearman"],
                "manhattan_pearson": raw_scores["manhattan"]["pearson"],
            }

        raw_scores = raw_scores.get(split, raw_scores)
        score_is_nested = isinstance(raw_scores[next(iter(raw_scores.keys()))], dict)
        if not score_is_nested:
            _scores: dict[str, dict[str, Union[float, str]]] = {lang: raw_scores for lang in self.languages}
            scores = _scores
        else:
            scores = raw_scores

        return scores

    def evaluate(self, model: Encoder) -> TaskResult:
        split = self.mteb_task.description["eval_splits"][0]
        # Infusing task into encode()
        original_encode = model.encode

        has_encode_queries = hasattr(model, "encode_queries")
        has_encode_corpus = hasattr(model, "encode_corpus")

        if has_encode_queries:
            original_encode_queries = model.encode_queries  # type: ignore
            model.encode_queries = partial(model.encode_queries, task=self)  # type: ignore
        if has_encode_corpus:
            original_encode_corpus = model.encode_corpus  # type: ignore
            model.encode_corpus = partial(model.encode_corpus, task=self)  # type: ignore

        try:
            model.encode = partial(model.encode, task=self)
            scores = self.mteb_task.evaluate(model, split=split)
        except Exception as e:
            raise e
        finally:
            # Resetting encode to original
            model.encode = original_encode
            if has_encode_queries:
                model.encode_queries = original_encode_queries  # type: ignore
            if has_encode_corpus:
                model.encode_corpus = original_encode_corpus  # type: ignore

        if scores is None:
            raise ValueError("MTEBTask evaluation failed.")

        # there is only one split in all MTEB tasks in SEB

        time_of_run: datetime = datetime.now()

        scores = self.format_scores(scores, split)

        task_result = TaskResult(
            task_name=self.name,
            task_description=self.description,
            task_version=self.version,
            time_of_run=time_of_run,
            scores=scores,  # type: ignore
            main_score=self.main_score,
        )

        return task_result
