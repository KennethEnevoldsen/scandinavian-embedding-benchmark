from __future__ import annotations

from datetime import date
from functools import partial
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from seb.interfaces.model import LazyLoadEncoder, ModelMeta, SebModel
from seb.registries import models


class BGEWrapper:
    """following the hf model card documentation."""

    def __init__(self, model_name: str, **kwargs: Any):  # noqa: ARG002
        self.model_name = model_name
        self.mdl = SentenceTransformer(model_name)
        self.sep = " "

    def to(self, device: torch.device) -> None:
        self.mdl.to(device)

    @staticmethod
    def reduce_max_len(sentences: list[str], max_len: int = 10_000) -> list[str]:
        _sentences = []
        for sent in sentences:
            _sentences.append(sent[:max_len])
        return _sentences

    def encode(  # type: ignore
        self,
        sentences: list[str],
        *,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> np.ndarray:
        if "task" in kwargs:
            kwargs.pop("task")

        return np.asarray(self.mdl.encode(self.reduce_max_len(sentences), batch_size=batch_size, **kwargs))

    def encode_queries(self, queries: list[str], batch_size: int = 32, **kwargs: Any) -> np.ndarray:
        if "task" in kwargs:
            kwargs.pop("task")
        sentences = ["Represent this sentence for searching relevant passages: " + sentence for sentence in queries]

        if "convert_to_tensor" in kwargs:
            kwargs.pop("convert_to_tensor")

        emb = self.mdl.encode(self.reduce_max_len(sentences), batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, **kwargs)
        return emb.astype("float16")  # type: ignore

    def encode_corpus(
        self,
        corpus: list[dict[str, str]] | dict[str, list[str]],
        batch_size: int = 32,
        **kwargs: Any,
    ) -> np.ndarray:
        if "task" in kwargs:
            kwargs.pop("task")
        if isinstance(corpus, dict):
            sentences = [
                (corpus["title"][i] + self.sep + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][i].strip()  # type: ignore
                for i in range(len(corpus["text"]))  # type: ignore
            ]
        else:
            sentences = [(doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus]
        if "convert_to_tensor" in kwargs:
            kwargs.pop("convert_to_tensor")

        emb = self.mdl.encode(self.reduce_max_len(sentences), batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, **kwargs)
        return emb.astype("float16")  # type: ignore


@models.register("bge-m3")
def create_bge_m3() -> SebModel:
    hf_name = "BAAI/bge-m3"
    meta = ModelMeta(
        name="bge-m3",
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=[],
        open_source=True,
        embedding_size=1024,
        architecture="XLM-R",
        release_date=date(2024, 5, 28),
    )
    return SebModel(
        encoder=LazyLoadEncoder(partial(BGEWrapper, model_name=hf_name)),  # type: ignore
        meta=meta,
    )


if __name__ == "__main__":
    model = create_bge_m3()
    test = model.encoder.encode(["Hello world", "test"])
    assert test.shape == (2, 1024)
