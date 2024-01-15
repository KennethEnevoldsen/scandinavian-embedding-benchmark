"""
The openai embedding api's evaluated on the SEB benchmark.
"""


import logging
from collections.abc import Sequence
from functools import partial

import torch

from seb.model_interface import EmbeddingModel, ModelInterface, ModelMeta
from seb.registries import models

logger = logging.getLogger(__name__)


class CohereTextEmbeddingModel(ModelInterface):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @staticmethod
    def create_sentence_blocks(
        sentences: Sequence[str],
        block_size: int,
    ) -> list[Sequence[str]]:
        sent_blocks: list[Sequence[str]] = []
        for i in range(0, len(sentences), block_size):
            sent_blocks.append(sentences[i : i + block_size])
        return sent_blocks

    def get_embedding_dim(self) -> int:
        v = self.encode(["get emb dim"])
        return v.shape[1]

    def encode(
        self,
        sentences: Sequence[str],
        batch_size: int = 32,  # noqa: ARG002
        embed_type: str = "classification",
        **kwargs: dict,  # noqa: ARG002
    ) -> torch.Tensor:
        import cohere  # type: ignore

        client = cohere.Client()
        response = client.embed(
            texts=list(sentences),
            model=self.model_name,
            input_type=embed_type,
        )

        return torch.tensor(response.embeddings)


@models.register("embed-multilingual-v3.0")
def create_embed_multilingual_v3() -> EmbeddingModel:
    model_name = "embed-multilingual-v3.0"
    meta = ModelMeta(
        name=model_name,
        huggingface_name=None,
        reference="https://huggingface.co/Cohere/Cohere-embed-multilingual-v3.0",
        languages=[],
        open_source=False,
        embedding_size=1024,
    )
    return EmbeddingModel(
        loader=partial(CohereTextEmbeddingModel, model_name=model_name),
        meta=meta,
    )
