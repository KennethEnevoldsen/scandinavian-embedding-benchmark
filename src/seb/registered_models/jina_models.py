from datetime import date
from functools import partial
from typing import Any, Literal, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from seb.interfaces.model import LazyLoadEncoder, ModelMeta, SebModel
from seb.interfaces.task import Task
from seb.registries import models

from .normalize_to_ndarray import normalize_to_ndarray
from .sentence_transformer_models import silence_warnings_from_sentence_transformers, wrap_sentence_transformer


class Jinav3EncoderWithTaskEncode(SentenceTransformer):
    """
    A sentence transformer wrapper that allows for encoding with a task.
    """

    def encode(  # type: ignore
        self,
        sentences: list[str],
        *,
        batch_size: int = 32,
        task: Optional[Task] = None,
        encode_type: Literal["query", "passage"] = "passage",
        **kwargs: Any,
    ) -> np.ndarray:
        task_prompt = None
        if task is not None:
            if task.task_type in ["STS", "BitextMining"]:
                task_prompt = "text-matching"
            if task.task_type in ["Classification"]:
                task_prompt = "classification"
            if task.task_type in ["Clustering"]:
                task_prompt = "separation"
            if task.task_type in ["Retrieval"] and encode_type == "query":
                task_prompt = "retrieval.query"
            if task.task_type in ["Retrieval"] and encode_type == "passage":
                task_prompt = "retrieval.passage"

        if task_prompt is None:
            emb = super().encode(sentences, batch_size=batch_size, **kwargs)
        else:
            emb = super().encode(sentences, batch_size=batch_size, task=task_prompt, prompt_name=task_prompt, **kwargs)
        return normalize_to_ndarray(emb)

    def encode_corpus(self, corpus: list[dict[str, str]], **kwargs: Any) -> np.ndarray:
        sep = " "
        if isinstance(corpus, dict):
            sentences = [
                (corpus["title"][i] + sep + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][i].strip()  # type: ignore
                for i in range(len(corpus["text"]))  # type: ignore
            ]
        else:
            sentences = [(doc["title"] + sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus]
        return self.encode(sentences, encode_type="passage", **kwargs)

    def encode_queries(self, queries: list[str], **kwargs: Any) -> np.ndarray:
        return self.encode(queries, encode_type="query", **kwargs)


def wrap_jina_sentence_transformer(model_name: str, max_seq_length: Optional[int] = None, **kwargs: Any) -> Jinav3EncoderWithTaskEncode:
    silence_warnings_from_sentence_transformers()
    mdl = Jinav3EncoderWithTaskEncode(model_name, **kwargs)
    if max_seq_length is not None:
        mdl.max_seq_length = max_seq_length
    return mdl


@models.register("jina-embeddings-v3")
def create_jina_embeddings_v3() -> SebModel:
    hf_name = "jinaai/jina-embeddings-v3"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=[],
        open_source=True,
        embedding_size=1024,
        architecture="XLM-R",
        release_date=date(2024, 8, 5),
    )
    return SebModel(
        encoder=LazyLoadEncoder(partial(wrap_jina_sentence_transformer, model_name=hf_name, trust_remote_code=True)),
        meta=meta,
    )


@models.register("jina-embedding-b-en-v1")
def create_jina_base() -> SebModel:
    hf_name = "jinaai/jina-embedding-b-en-v1"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["en"],
        open_source=True,
        embedding_size=768,
        architecture="T5",
        release_date=date(2023, 7, 7),
    )
    return SebModel(
        encoder=LazyLoadEncoder(partial(wrap_sentence_transformer, model_name=hf_name)),  # type: ignore
        meta=meta,
    )
