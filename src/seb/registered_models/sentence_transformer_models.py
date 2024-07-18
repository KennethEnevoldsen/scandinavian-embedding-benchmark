"""
All the models registered in the benchmark, along with their metadata.
"""

import logging
from datetime import date
from functools import partial
from typing import Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from seb.interfaces.model import LazyLoadEncoder, ModelMeta, SebModel
from seb.interfaces.task import Task
from seb.registries import models

from .normalize_to_ndarray import normalize_to_ndarray


def silence_warnings_from_sentence_transformers():
    from sentence_transformers.SentenceTransformer import logger

    logger.setLevel(logging.ERROR)


class SentenceTransformerWithTaskEncode(SentenceTransformer):
    """
    A sentence transformer wrapper that allows for encoding with a task.
    """

    def encode(  # type: ignore
        self,
        sentences: list[str],
        *,
        batch_size: int = 32,
        task: Optional[Task] = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> np.ndarray:
        emb = super().encode(sentences, batch_size=batch_size, **kwargs)
        return normalize_to_ndarray(emb)


def wrap_sentence_transformer(
    model_name: str,
    max_seq_length: Optional[int] = None,
) -> SentenceTransformerWithTaskEncode:
    silence_warnings_from_sentence_transformers()
    mdl = SentenceTransformerWithTaskEncode(model_name)
    if max_seq_length is not None:
        mdl.max_seq_length = max_seq_length
    return mdl


@models.register("jina-embedding-b-en-v1")
def create_jina_base() -> SebModel:
    hf_name = "jinaai/jina-embedding-b-en-v1"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["en"],
        open_source=True,
        embedding_size=768,
        architecture="T5",
        release_date=date(2023, 7, 7),
    )
    return SebModel(
        encoder=LazyLoadEncoder(partial(wrap_sentence_transformer, model_name=hf_name)),  # type: ignore
        meta=meta,
    )


# Relevant multilingual models
@models.register("all-MiniLM-L6-v2")
def create_all_mini_lm_l6_v2() -> SebModel:
    hf_name = "sentence-transformers/all-MiniLM-L6-v2"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["en"],
        open_source=True,
        embedding_size=384,
        architecture="BERT",
        release_date=date(2021, 6, 30),
    )
    return SebModel(
        encoder=LazyLoadEncoder(partial(wrap_sentence_transformer, model_name=hf_name)),  # type: ignore
        meta=meta,
    )


@models.register("paraphrase-multilingual-MiniLM-L12-v2")
def create_multilingual_mini_lm_l12_v2() -> SebModel:
    hf_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=[],
        open_source=True,
        embedding_size=384,
        architecture="BERT",
        release_date=date(2021, 6, 2),
    )
    return SebModel(
        encoder=LazyLoadEncoder(partial(wrap_sentence_transformer, model_name=hf_name)),  # type: ignore
        meta=meta,
    )


@models.register("paraphrase-multilingual-mpnet-base-v2")
def create_multilingual_mpnet_base_v2() -> SebModel:
    hf_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=[],
        open_source=True,
        embedding_size=768,
        architecture="BERT",
        release_date=date(2021, 6, 2),
    )
    return SebModel(
        encoder=LazyLoadEncoder(partial(wrap_sentence_transformer, model_name=hf_name)),  # type: ignore
        meta=meta,
    )


@models.register("sentence-bert-swedish-cased")
def create_sentence_swedish_cased() -> SebModel:
    hf_name = "KBLab/sentence-bert-swedish-cased"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["sv"],
        open_source=True,
        embedding_size=768,
        architecture="BERT",
        release_date=date(2021, 8, 8),
    )
    return SebModel(
        encoder=LazyLoadEncoder(partial(wrap_sentence_transformer, model_name=hf_name)),  # type: ignore
        meta=meta,
    )


@models.register("electra-small-nordic")
def create_electra_small_nordic() -> SebModel:
    hf_name = "jonfd/electra-small-nordic"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["da", "nb", "sv", "nn"],
        open_source=True,
        embedding_size=256,
        architecture="ELECTRA",
        release_date=date(2022, 1, 31),
    )
    return SebModel(
        encoder=LazyLoadEncoder(partial(wrap_sentence_transformer, model_name=hf_name)),  # type: ignore
        meta=meta,
    )


@models.register("DanskBERT")
def create_dansk_bert() -> SebModel:
    hf_name = "vesteinn/DanskBERT"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["da"],
        open_source=True,
        embedding_size=768,
        architecture="XLM-R",
        release_date=date(2022, 11, 23),
    )

    return SebModel(
        # see https://huggingface.co/vesteinn/DanskBERT/discussions/3
        encoder=LazyLoadEncoder(partial(wrap_sentence_transformer, model_name=hf_name, max_seq_length=512)),  # type: ignore
        meta=meta,
    )


@models.register("dfm-encoder-large-v1")
def create_dfm_encoder_large_v1() -> SebModel:
    hf_name = "chcaa/dfm-encoder-large-v1"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["da"],
        open_source=True,
        embedding_size=1024,
        release_date=date(2023, 1, 4),
    )
    return SebModel(
        encoder=LazyLoadEncoder(partial(wrap_sentence_transformer, model_name=hf_name)),  # type: ignore
        meta=meta,
    )


@models.register("nb-bert-large")
def create_nb_bert_large() -> SebModel:
    hf_name = "NbAiLab/nb-bert-large"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["nb", "nn"],
        open_source=True,
        embedding_size=1024,
        architecture="BERT",
        release_date=date(2021, 4, 29),
    )
    return SebModel(
        encoder=LazyLoadEncoder(partial(wrap_sentence_transformer, model_name=hf_name)),  # type: ignore
        meta=meta,
    )


@models.register("nb-bert-base")
def create_nb_bert_base() -> SebModel:
    hf_name = "NbAiLab/nb-bert-base"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["nb", "nn"],
        open_source=True,
        embedding_size=768,
        architecture="BERT",
        release_date=date(2021, 1, 13),
    )
    return SebModel(
        encoder=LazyLoadEncoder(partial(wrap_sentence_transformer, model_name=hf_name)),  # type: ignore
        meta=meta,
    )


"""
# excluded due to loading issues: https://huggingface.co/ltg/norbert3-base/discussions/1
# I could create a custom encoder for this model

@models.register("ltg/norbert3-large")
def create_norbert3_large() -> odel:
    hf_name = "ltg/norbert3-large"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["nb", "nn"],
    )
    return odel(
        encoder=LazyLoadEncoder(partial(get_sentence_transformer, model_name=hf_name),  # type: ignore
        meta=meta,
    )


@models.register("ltg/norbert3-base")
def create_norbert3_base() -> odel:
    hf_name = "ltg/norbert3-base"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["nb", "nn"],
    )
    return odel(
        encoder=LazyLoadEncoder(partial(get_sentence_transformer, model_name=hf_name),  # type: ignore
        meta=meta,
    )
"""


@models.register("bert-base-swedish-cased")
def create_bert_base_swedish_cased() -> SebModel:
    hf_name = "KB/bert-base-swedish-cased"

    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["sv"],
        open_source=True,
        embedding_size=768,
        architecture="BERT",
        release_date=date(2022, 6, 7),
    )
    return SebModel(
        encoder=LazyLoadEncoder(partial(wrap_sentence_transformer, model_name=hf_name, max_seq_length=512)),  # type: ignore
        meta=meta,
    )


@models.register("electra-small-swedish-cased-discriminator")
def create_electra_small_swedish_cased_discriminator() -> SebModel:
    hf_name = "kb/electra-small-swedish-cased-discriminator"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["sv"],
        open_source=True,
        embedding_size=256,
        architecture="ELECTRA",
        release_date=date(2022, 6, 7),
    )
    return SebModel(
        encoder=LazyLoadEncoder(partial(wrap_sentence_transformer, model_name=hf_name)),  # type: ignore
        meta=meta,
    )


# Multilingual baselines
@models.register("xlm-roberta-base")
def create_xlm_roberta_base() -> SebModel:
    hf_name = "xlm-roberta-base"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        open_source=True,
        embedding_size=768,
        release_date=date(2019, 6, 29),
    )

    # Beware that this uses mean pooling currently, and we might want to change it to CLS in the future
    return SebModel(
        encoder=LazyLoadEncoder(partial(wrap_sentence_transformer, model_name=hf_name, max_seq_length=512)),  # type: ignore
        meta=meta,
    )


@models.register("xlm-roberta-large")
def create_xlm_roberta_large() -> SebModel:
    hf_name = "xlm-roberta-large"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        open_source=True,
        embedding_size=1024,
        architecture="XLM-R",
    )

    # Beware that this uses mean pooling currently, and we might want to change it to CLS in the future
    return SebModel(
        encoder=LazyLoadEncoder(partial(wrap_sentence_transformer, model_name=hf_name)),  # type: ignore
        meta=meta,
    )


@models.register("LaBSE")
def create_labse() -> SebModel:
    hf_name = "sentence-transformers/LaBSE"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        open_source=True,
        embedding_size=768,
        architecture="BERT",
        release_date=date(2020, 10, 12),
    )

    return SebModel(
        encoder=LazyLoadEncoder(partial(wrap_sentence_transformer, model_name=hf_name)),  # type: ignore
        meta=meta,
    )


# Scandinavian sentence encoders
@models.register("dfm-encoder-large-v1 (SimCSE)")
def create_dfm_sentence_encoder_large() -> SebModel:
    hf_name = "KennethEnevoldsen/dfm-sentence-encoder-large-1"
    meta = ModelMeta(
        name="dfm-encoder-large-v1 (SimCSE)",
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["da"],
        open_source=True,
        embedding_size=1024,
        architecture="BERT",
        release_date=date(2023, 11, 15),
    )
    return SebModel(
        encoder=LazyLoadEncoder(partial(wrap_sentence_transformer, model_name=hf_name)),  # type: ignore
        meta=meta,
    )


@models.register("dfm-sentence-encoder-large-exp1")
def create_dfm_sentence_encoder_large_exp() -> SebModel:
    hf_name = "KennethEnevoldsen/dfm-sentence-encoder-large-exp1"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["da"],
        open_source=True,
        embedding_size=1024,
        release_date=date(2023, 11, 15),
    )
    return SebModel(
        encoder=LazyLoadEncoder(partial(wrap_sentence_transformer, model_name=hf_name)),  # type: ignore
        meta=meta,
    )


@models.register("dfm-encoder-small-v1 (SimCSE)")
def create_dfm_sentence_encoder_small() -> SebModel:
    hf_name = "KennethEnevoldsen/dfm-sentence-encoder-small-v1"
    meta = ModelMeta(
        name="dfm-encoder-small-v1 (SimCSE)",
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["da"],
        open_source=True,
        embedding_size=256,
    )
    return SebModel(
        encoder=LazyLoadEncoder(partial(wrap_sentence_transformer, model_name=hf_name)),  # type: ignore
        meta=meta,
    )


@models.register("dfm-sentence-encoder-medium (SimCSE)")
def create_dfm_sentence_encoder_medium() -> SebModel:
    hf_name = "KennethEnevoldsen/dfm-sentence-encoder-medium-v1"
    meta = ModelMeta(
        name="dfm-sentence-encoder-medium (SimCSE)",
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["da"],
        open_source=True,
        embedding_size=768,
        architecture="BERT",
        release_date=date(2023, 11, 15),
    )
    return SebModel(
        encoder=LazyLoadEncoder(partial(wrap_sentence_transformer, model_name=hf_name)),  # type: ignore
        meta=meta,
    )


@models.register("dfm-sentence-encoder-large-exp2-no-lang-align")
def create_dfm_sentence_encoder_large_exp2() -> SebModel:
    hf_name = "KennethEnevoldsen/dfm-sentence-encoder-large-exp2-no-lang-align"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["da"],
        open_source=True,
        embedding_size=1024,
        architecture="BERT",
        release_date=date(2023, 11, 15),
    )
    return SebModel(
        encoder=LazyLoadEncoder(partial(wrap_sentence_transformer, model_name=hf_name)),  # type: ignore
        meta=meta,
    )


@models.register("mxbai-embed-large-v1")
def create_mxbai_embed_large_v1() -> SebModel:
    hf_name = "mixedbread-ai/mxbai-embed-large-v1"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["da"],
        open_source=True,
        embedding_size=1024,
        architecture="BERT",
        release_date=date(2024, 4, 12),
    )
    return SebModel(
        encoder=LazyLoadEncoder(partial(wrap_sentence_transformer, model_name=hf_name)),  # type: ignore
        meta=meta,
    )


@models.register("use-cmlm-multilingual")
def create_use_cmlm_multilingual() -> SebModel:
    hf_name = "sentence-transformers/use-cmlm-multilingual"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        open_source=True,
        embedding_size=768,
        architecture="BERT",
        release_date=date(2022, 4, 14),
    )
    return SebModel(
        encoder=LazyLoadEncoder(partial(wrap_sentence_transformer, model_name=hf_name)),  # type: ignore
        meta=meta,
    )


if __name__ == "__main__":
    import seb

    model = seb.get_model("mxbai-embed-large-v1")
    test = model.encoder.encode(["Hello world", "test"])
    test.shape
