"""
The openai embedding api's evaluated on the SEB benchmark.
"""

import logging
from collections.abc import Sequence
from datetime import date
from functools import partial
from typing import Any

import torch

from seb.interfaces.model import Encoder, LazyLoadEncoder, ModelMeta, SebModel
from seb.interfaces.task import Task
from seb.registries import models

logger = logging.getLogger(__name__)


class OpenaiTextEmbeddingModel(Encoder):
    def __init__(self, api_name: str, input_sentences: int = 64) -> None:
        self.api_name = api_name
        self.input_sentences = input_sentences

    @staticmethod
    def preprocess(sentences: Sequence[str]) -> Sequence[str]:
        return [t.replace("\n", " ") for t in sentences]

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
        v = self.embed(["get emb dim"])
        return v.shape[1]

    def embed(self, sentences: Sequence[str]) -> torch.Tensor:
        import openai  # type: ignore
        from openai.error import InvalidRequestError  # type: ignore

        try:
            emb = openai.Embedding.create(  # type: ignore
                input=sentences,
                model=self.api_name,
            )
        except InvalidRequestError as e:
            if "Please reduce your prompt" in e._message:  # type: ignore
                if len(sentences) == 1:
                    logger.warning("Text is too long for the API, truncating.")
                    max_size_in_str_text = sentences[0][:50_000]
                    half = len(max_size_in_str_text) // 2
                    first_half = max_size_in_str_text[:half]
                    return self.embed([first_half])

                half = len(sentences) // 2
                return torch.cat(
                    [
                        self.embed(sentences[:half]),
                        self.embed(sentences[half:]),
                    ],
                )
            raise e
        data = emb["data"]  # type: ignore
        vectors = [embedding.embedding for embedding in data]
        return torch.tensor(vectors)

    def encode(  # type: ignore
        self,
        sentences: Sequence[str],
        *,
        task: "Task",  # noqa: ARG002
        batch_size: int = 32,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> torch.Tensor:
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
        open_source=False,
        embedding_size=1536,
        architecture="API",
        release_date=date(2022, 1, 25),
    )
    return SebModel(
        encoder=LazyLoadEncoder(partial(OpenaiTextEmbeddingModel, api_name=api_name)),
        meta=meta,
    )


@models.register("text-embedding-3-small")
def create_openai_3_small() -> SebModel:
    api_name = "text-embedding-3-small"
    meta = ModelMeta(
        name=api_name,
        huggingface_name=None,
        reference="https://openai.com/blog/new-and-improved-embedding-model",
        languages=[],
        open_source=False,
        embedding_size=1536,
        architecture="API",
        release_date=date(2024, 1, 25),
    )
    return SebModel(
        encoder=LazyLoadEncoder(partial(OpenaiTextEmbeddingModel, api_name=api_name)),
        meta=meta,
    )


@models.register("text-embedding-3-large")
def create_openai_3_large() -> SebModel:
    api_name = "text-embedding-3-large"
    meta = ModelMeta(
        name=api_name,
        huggingface_name=None,
        reference="https://openai.com/blog/new-and-improved-embedding-model",
        languages=[],
        open_source=False,
        embedding_size=3072,
        architecture="API",
        release_date=date(2024, 1, 25),
    )
    return SebModel(
        encoder=LazyLoadEncoder(partial(OpenaiTextEmbeddingModel, api_name=api_name)),
        meta=meta,
    )
