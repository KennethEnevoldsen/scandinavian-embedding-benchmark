"""
The openai embedding api's evaluated on the SEB benchmark.
"""


import logging
from functools import partial
from typing import Sequence

import torch

from seb.model_interface import ModelInterface, ModelMeta, SebModel
from seb.registries import models

logger = logging.getLogger(__name__)


class OpenaiTextEmbeddingModel(ModelInterface):
    def __init__(self, api_name: str, input_sentences: int = 64) -> None:
        self.api_name = api_name
        self.input_sentences = input_sentences

    @staticmethod
    def preprocess(sentences: Sequence[str]) -> Sequence[str]:
        return [t.replace("\n", " ") for t in sentences]

    @staticmethod
    def create_sentence_blocks(
        sentences: Sequence[str], block_size: int
    ) -> list[Sequence[str]]:
        sent_blocks: list[Sequence[str]] = []
        for i in range(0, len(sentences), block_size):
            sent_blocks.append(sentences[i : i + block_size])
        return sent_blocks

    def get_embedding_dim(self) -> int:
        v = self.embed(["get emb dim"])
        return v.shape[1]

    @staticmethod
    def embed(sentences: Sequence[str]) -> torch.Tensor:
        import openai
        from openai.error import InvalidRequestError

        try:
            emb = openai.Embedding.create(
                input=sentences, model="text-embedding-ada-002"
            )
        except InvalidRequestError as e:
            if "Please reduce your prompt" in e._message:  # type: ignore
                if len(sentences) == 1:
                    logger.warning("Text is too long for the API, truncating.")
                    max_size_in_str_text = sentences[0][:50_000]
                    half = len(max_size_in_str_text) // 2
                    first_half = max_size_in_str_text[:half]
                    return OpenaiTextEmbeddingModel.embed([first_half])

                half = len(sentences) // 2
                return torch.cat(
                    [
                        OpenaiTextEmbeddingModel.embed(sentences[:half]),
                        OpenaiTextEmbeddingModel.embed(sentences[half:]),
                    ]
                )
            else:
                raise e
        data = emb["data"]  # type: ignore
        vectors = [embedding.embedding for embedding in data]
        return torch.tensor(vectors)

    def encode(
        self,
        sentences: Sequence[str],
        batch_size: int = 32,  # noqa: ARG002
        **kwargs: dict,  # noqa: ARG002
    ) -> torch.Tensor:
        import openai

        sentences = self.preprocess(sentences)
        sent_blocks = self.create_sentence_blocks(sentences, self.input_sentences)

        vectors = []

        for sent_block in sent_blocks:
            vectors.append(self.embed(sent_block))

        return torch.cat(vectors)


@models.register("text-embedding-ada-002")
def create_openai_ada_002() -> SebModel:
    api_name = "text-embedding-ada-002"
    meta = ModelMeta(
        name=api_name,
        huggingface_name=None,
        reference="https://openai.com/blog/new-and-improved-embedding-model",
        languages=[],
    )
    return SebModel(
        loader=partial(OpenaiTextEmbeddingModel, api_name=api_name),
        meta=meta,
    )
