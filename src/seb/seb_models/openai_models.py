"""
The openai embedding api's evaluated on the SEB benchmark.
"""

from functools import partial
from typing import List

import torch

from seb.model_interface import ModelInterface, ModelMeta, SebModel
from seb.registries import models


class OpenaiTextEmbeddingModel(ModelInterface):
    def __init__(self, api_name: str):
        self.api_name = api_name

    def encode(self, sentences: List[str], batch_size: int = 32) -> torch.Tensor:
        import openai

        sentences = [t.replace("\n", " ") for t in sentences]
        emb = openai.Embedding.create(input=sentences, model=self.api_name)
        data = emb["data"]
        vectors = [embedding.embedding for embedding in data]
        return torch.tensor(vectors)


@models.register("text-embedding-ada-002")
def create_openai_ada_002() -> SebModel:
    api_name = "text-embedding-ada-002"
    meta = ModelMeta(
        name=api_name,
        huggingface_name=None,
        reference=f"https://openai.com/blog/new-and-improved-embedding-model",
        languages=[],
    )
    return SebModel(
        loader=partial(OpenaiTextEmbeddingModel, api_name=api_name),
        meta=meta,
    )
