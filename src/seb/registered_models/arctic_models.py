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


class ArcticEncoderWithTaskEncode(SentenceTransformer):
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
        if task is not None and task.task_type in ["Retrieval"] and encode_type == "query":
            task_prompt = "query"

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


def wrap_arctic_sentence_transformer(model_name: str, max_seq_length: Optional[int] = None, **kwargs: Any) -> ArcticEncoderWithTaskEncode:
    silence_warnings_from_sentence_transformers()
    mdl = ArcticEncoderWithTaskEncode(model_name, **kwargs)
    if max_seq_length is not None:
        mdl.max_seq_length = max_seq_length
    return mdl


@models.register("snowflake-arctic-embed-l-v2.0")
def create_artic_embed_l_v2() -> SebModel:
    hf_name = "Snowflake/snowflake-arctic-embed-l-v2.0"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=[],
        open_source=True,
        embedding_size=1024,
        architecture="XLM-R",
        release_date=date(2024, 11, 8),
    )
    return SebModel(
        encoder=LazyLoadEncoder(partial(wrap_arctic_sentence_transformer, model_name=hf_name, trust_remote_code=True)),  # type: ignore
        meta=meta,
    )


@models.register("snowflake-arctic-embed-m-v2.0")
def create_artic_embed_m_v2() -> SebModel:
    hf_name = "Snowflake/snowflake-arctic-embed-m-v2.0"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=[],
        open_source=True,
        embedding_size=768,
        architecture="GTE",
        release_date=date(2024, 11, 8),
    )
    return SebModel(
        encoder=LazyLoadEncoder(partial(wrap_arctic_sentence_transformer, model_name=hf_name, trust_remote_code=True)),  # type: ignore
        meta=meta,
    )
