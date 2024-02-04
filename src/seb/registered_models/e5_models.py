from functools import partial
from typing import Any, Optional

from numpy.typing import ArrayLike
from sentence_transformers import SentenceTransformer

from seb.registries import models

from ..interfaces.model import Encoder, LazyLoadEncoder, ModelMeta, SebModel
from ..interfaces.task import Task


class E5Wrapper(Encoder):
    def __init__(self, model_name: str, sep: str = " "):
        self.model_name = model_name
        self.mdl = SentenceTransformer(model_name)
        self.sep = sep

    def encode(  # type: ignore
        self,
        sentences: list[str],
        *,
        task: Task,  # noqa: ARG002
        batch_size: int = 32,
        **kwargs: Any,
    ) -> ArrayLike:
        return self.encode_queries(sentences, batch_size=batch_size, **kwargs)

    def encode_queries(self, queries: list[str], batch_size: int, **kwargs):  # noqa
        sentences = ["query: " + sentence for sentence in queries]
        return self.mdl.encode(sentences, batch_size=batch_size, **kwargs)

    def encode_corpus(self, corpus: list[dict[str, str]], batch_size: int, **kwargs):  # noqa
        if isinstance(corpus, dict):
            sentences = [
                (corpus["title"][i] + self.sep + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][i].strip()  # type: ignore
                for i in range(len(corpus["text"]))  # type: ignore
            ]
        else:
            sentences = [(doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus]
        sentences = ["passage: " + sentence for sentence in sentences]
        return self.mdl.encode(sentences, batch_size=batch_size, **kwargs)


# English
@models.register("intfloat/e5-small")
def create_e5_small() -> SebModel:
    hf_name = "intfloat/e5-small"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["en"],
        open_source=True,
        embedding_size=384,
    )
    return SebModel(
        encoder=LazyLoadEncoder(partial(E5Wrapper, model_name=hf_name)),
        meta=meta,
    )


@models.register("intfloat/e5-base")
def create_e5_base() -> SebModel:
    hf_name = "intfloat/e5-base"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["en"],
        open_source=True,
        embedding_size=768,
    )
    return SebModel(
        encoder=LazyLoadEncoder(partial(E5Wrapper, model_name=hf_name)),
        meta=meta,
    )


@models.register("intfloat/e5-large")
def create_e5_large() -> SebModel:
    hf_name = "intfloat/e5-large"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["en"],
        open_source=True,
        embedding_size=1024,
    )
    return SebModel(
        encoder=LazyLoadEncoder(partial(E5Wrapper, model_name=hf_name)),
        meta=meta,
    )


# Multilingual
@models.register("intfloat/multilingual-e5-small")
def create_multilingual_e5_small() -> SebModel:
    hf_name = "intfloat/multilingual-e5-small"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=[],
        open_source=True,
        embedding_size=384,
    )
    return SebModel(
        encoder=LazyLoadEncoder(partial(E5Wrapper, model_name=hf_name)),
        meta=meta,
    )


@models.register("intfloat/multilingual-e5-base")
def create_multilingual_e5_base() -> SebModel:
    hf_name = "intfloat/multilingual-e5-base"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=[],
        open_source=True,
        embedding_size=768,
    )
    return SebModel(
        encoder=LazyLoadEncoder(partial(E5Wrapper, model_name=hf_name)),
        meta=meta,
    )


@models.register("intfloat/multilingual-e5-large")
def create_multilingual_e5_large() -> SebModel:
    hf_name = "intfloat/multilingual-e5-large"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=[],
        open_source=True,
        embedding_size=1024,
    )
    return SebModel(
        encoder=LazyLoadEncoder(partial(E5Wrapper, model_name=hf_name)),
        meta=meta,
    )
