from collections.abc import Sequence
from functools import partial

import numpy as np
import torch

import seb
from seb.registries import models


class FastTextModel(seb.Encoder):
    def __init__(self, model_name: str, lang: str) -> None:
        self.model_name = model_name
        self.lang = lang

    def get_embedding_dim(self) -> int:
        v = self.encode(["get emb dim"])
        return v.shape[1]

    def encode(
        self,
        sentences: Sequence[str],
        **kwargs: dict,  # noqa: ARG002
    ) -> torch.Tensor:
        import fasttext
        import fasttext.util

        fasttext.util.download_model(self.lang, if_exists="ignore")
        model = fasttext.load_model(self.model_name)
        embeddings = []
        for sentence in sentences:
            # This is to appease FastText as they made the function err
            # if there's a \n in the sentence.
            sentence = " ".join(sentence.split())
            sentence_embedding = model.get_sentence_vector(sentence.strip())
            embeddings.append(sentence_embedding)
        return torch.tensor(np.stack(embeddings))


@models.register("fasttext-cc-da-300")
def create_cc_da_300() -> seb.EmbeddingModel:
    model_name = "fasttext-cc-da-300"
    meta = seb.ModelMeta(
        name=model_name,
        huggingface_name=None,
        reference="https://fasttext.cc/docs/en/crawl-vectors.html",
        languages=["da"],
        open_source=True,
        embedding_size=300,
    )
    return seb.EmbeddingModel(
        loader=partial(FastTextModel, model_name="cc.da.300.bin", lang="da"),
        meta=meta,
    )


@models.register("fasttext-cc-sv-300")
def create_cc_sv_300() -> seb.EmbeddingModel:
    model_name = "fasttext-cc-sv-300"
    meta = seb.ModelMeta(
        name=model_name,
        huggingface_name=None,
        reference="https://fasttext.cc/docs/en/crawl-vectors.html",
        languages=["sv"],
        open_source=True,
        embedding_size=300,
    )
    return seb.EmbeddingModel(
        loader=partial(FastTextModel, model_name="cc.sv.300.bin", lang="sv"),
        meta=meta,
    )


@models.register("fasttext-cc-nb-300")
def create_cc_nb_300() -> seb.EmbeddingModel:
    model_name = "fasttext-cc-nb-300"
    meta = seb.ModelMeta(
        name=model_name,
        huggingface_name=None,
        reference="https://fasttext.cc/docs/en/crawl-vectors.html",
        languages=["nb"],
        open_source=True,
        embedding_size=300,
    )
    return seb.EmbeddingModel(
        loader=partial(FastTextModel, model_name="cc.no.300.bin", lang="no"),
        meta=meta,
    )


@models.register("fasttext-cc-nn-300")
def create_cc_nn_300() -> seb.EmbeddingModel:
    model_name = "fasttext-cc-nn-300"
    meta = seb.ModelMeta(
        name=model_name,
        huggingface_name=None,
        reference="https://fasttext.cc/docs/en/crawl-vectors.html",
        languages=["nn"],
        open_source=True,
        embedding_size=300,
    )
    return seb.EmbeddingModel(
        loader=partial(FastTextModel, model_name="cc.nn.300.bin", lang="nn"),
        meta=meta,
    )
