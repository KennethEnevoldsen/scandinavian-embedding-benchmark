"""
The openai embedding api's evaluated on the SEB benchmark.
"""


import logging
from datetime import date
from functools import partial
from typing import Any, Optional

import torch

from seb.interfaces.model import Encoder, LazyLoadEncoder, ModelMeta, SebModel
from seb.interfaces.task import Task
from seb.registries import models

logger = logging.getLogger(__name__)


class CohereTextEmbeddingModel(Encoder):
    def __init__(self, model_name: str, sep: str = " ") -> None:
        self.model_name = model_name
        self.sep = sep

    def get_embedding_dim(self) -> int:
        v = self._embed(["get emb dim"], input_type="classification")
        return v.shape[1]

    def _embed(self, sentences: list[str], input_type: str) -> torch.Tensor:
        import cohere

        client = cohere.Client()
        response = client.embed(
            texts=list(sentences),
            model=self.model_name,
            input_type=input_type,
        )
        return torch.tensor(response.embeddings)

    def encode(
        self,
        sentences: list[str],
        batch_size: int = 32,  # noqa: ARG002
        *,
        task: Optional[Task] = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> torch.Tensor:
        if task and task.task_type == "Classification":
            input_type = "classification"
        elif task and task.task_type == "Clustering":
            input_type = "clustering"
        else:
            input_type = "search_document"
        return self._embed(sentences, input_type=input_type)

    def encode_queries(self, queries: list[str], batch_size: int, **kwargs):  # noqa
        return self._embed(queries, input_type="search_query")

    def encode_corpus(self, corpus: list[dict[str, str]], batch_size: int, **kwargs):  # noqa
        if isinstance(corpus, dict):
            sentences = [
                (corpus["title"][i] + self.sep + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][i].strip()  # type: ignore
                for i in range(len(corpus["text"]))  # type: ignore
            ]
        else:
            sentences = [(doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus]
        return self._embed(sentences, input_type="search_document")


@models.register("embed-multilingual-v3.0")
def create_embed_multilingual_v3() -> SebModel:
    model_name = "embed-multilingual-v3.0"
    meta = ModelMeta(
        name=model_name,
        huggingface_name=None,
        reference="https://huggingface.co/Cohere/Cohere-embed-multilingual-v3.0",
        languages=[],
        open_source=False,
        embedding_size=1024,
        model_type="API",
        release_date=date(2023, 11, 2),
    )
    return SebModel(
        encoder=LazyLoadEncoder(partial(CohereTextEmbeddingModel, model_name=model_name)),
        meta=meta,
    )
