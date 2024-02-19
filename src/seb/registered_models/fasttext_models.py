from collections.abc import Sequence
from datetime import date
from functools import partial
from typing import Any

import numpy as np
import torch

from seb.interfaces.model import Encoder, LazyLoadEncoder, ModelMeta, SebModel
from seb.registries import models


class FastTextModel(Encoder):
    def __init__(self, model_name: str, lang: str) -> None:
        self.model_name = model_name
        self.lang = lang

        import fasttext  # type: ignore
        import fasttext.util  # type: ignore

        fasttext.util.download_model(self.lang, if_exists="ignore")
        self.model = fasttext.load_model(self.model_name)

    def get_embedding_dim(self) -> int:
        v = self.encode(["get emb dim"])
        return v.shape[1]

    def encode(  # type: ignore
        self,
        sentences: Sequence[str],
        **kwargs: Any,  # noqa: ARG002
    ) -> torch.Tensor:
        embeddings = []
        for sentence in sentences:
            # This is to appease FastText as they made the function err
            # if there's a \n in the sentence.
            sentence = " ".join(sentence.split())  # noqa
            sentence_embedding = self.model.get_sentence_vector(sentence)
            embeddings.append(sentence_embedding)
        return torch.tensor(np.stack(embeddings))


@models.register("fasttext-cc-da-300")
def create_cc_da_300() -> SebModel:
    model_name = "fasttext-cc-da-300"
    meta = ModelMeta(
        name=model_name,
        huggingface_name=None,
        reference="https://fasttext.cc/docs/en/crawl-vectors.html",
        languages=["da"],
        open_source=True,
        embedding_size=300,
        architecture="fastText",
        release_date=date(2017, 1, 1),
    )
    return SebModel(
        encoder=LazyLoadEncoder(partial(FastTextModel, model_name="cc.da.300.bin", lang="da")),
        meta=meta,
    )


@models.register("fasttext-cc-sv-300")
def create_cc_sv_300() -> SebModel:
    model_name = "fasttext-cc-sv-300"
    meta = ModelMeta(
        name=model_name,
        huggingface_name=None,
        reference="https://fasttext.cc/docs/en/crawl-vectors.html",
        languages=["sv"],
        open_source=True,
        embedding_size=300,
        architecture="fastText",
        release_date=date(2017, 1, 1),
    )
    return SebModel(
        encoder=LazyLoadEncoder(partial(FastTextModel, model_name="cc.sv.300.bin", lang="sv")),
        meta=meta,
    )


@models.register("fasttext-cc-nb-300")
def create_cc_nb_300() -> SebModel:
    model_name = "fasttext-cc-nb-300"
    meta = ModelMeta(
        name=model_name,
        huggingface_name=None,
        reference="https://fasttext.cc/docs/en/crawl-vectors.html",
        languages=["nb"],
        open_source=True,
        embedding_size=300,
        architecture="fastText",
        release_date=date(2017, 1, 1),
    )
    return SebModel(
        encoder=LazyLoadEncoder(partial(FastTextModel, model_name="cc.no.300.bin", lang="no")),
        meta=meta,
    )


@models.register("fasttext-cc-nn-300")
def create_cc_nn_300() -> SebModel:
    model_name = "fasttext-cc-nn-300"
    meta = ModelMeta(
        name=model_name,
        huggingface_name=None,
        reference="https://fasttext.cc/docs/en/crawl-vectors.html",
        languages=["nn"],
        open_source=True,
        embedding_size=300,
        architecture="fastText",
        release_date=date(2017, 1, 1),
    )
    return SebModel(
        encoder=LazyLoadEncoder(partial(FastTextModel, model_name="cc.nn.300.bin", lang="nn")),
        meta=meta,
    )
