"""
All the models registered in the benchmark, along with their metadata.
"""
import logging
from functools import partial
from typing import Any, Optional

from sentence_transformers import SentenceTransformer

from seb.interfaces.model import EmbeddingModel, ModelMeta
from seb.interfaces.task import Task
from seb.registries import models

from ..types import ArrayLike


def silence_warnings_from_sentence_transformers():
    from sentence_transformers.SentenceTransformer import logger

    logger.setLevel(logging.ERROR)


class SentenceTransformerWithTaskEncode(SentenceTransformer):
    """
    A sentence transformer wrapper that allows for encoding with a task.
    """

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        task: Task,  # noqa: ARG002
        **kwargs: Any,
    ) -> ArrayLike:
        return super().encode(sentences, batch_size=batch_size, **kwargs)  # type: ignore


def get_sentence_transformer(
    model_name: str,
    max_seq_length: Optional[int] = None,
) -> SentenceTransformerWithTaskEncode:
    silence_warnings_from_sentence_transformers()
    mdl = SentenceTransformerWithTaskEncode(model_name)
    if max_seq_length is not None:
        mdl.max_seq_length = max_seq_length
    return mdl


# Relevant multilingual models
@models.register("sentence-transformers/all-MiniLM-L6-v2")
def create_all_mini_lm_l6_v2() -> EmbeddingModel:
    hf_name = "sentence-transformers/all-MiniLM-L6-v2"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["en"],
        open_source=True,
        embedding_size=384,
    )
    return EmbeddingModel(
        loader=partial(get_sentence_transformer, model_name=hf_name),  # type: ignore
        meta=meta,
    )


@models.register("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
def create_multilingual_mini_lm_l12_v2() -> EmbeddingModel:
    hf_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=[],
        open_source=True,
        embedding_size=384,
    )
    return EmbeddingModel(
        loader=partial(get_sentence_transformer, model_name=hf_name),  # type: ignore
        meta=meta,
    )


@models.register("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
def create_multilingual_mpnet_base_v2() -> EmbeddingModel:
    hf_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=[],
        open_source=True,
        embedding_size=768,
    )
    return EmbeddingModel(
        loader=partial(get_sentence_transformer, model_name=hf_name),  # type: ignore
        meta=meta,
    )


@models.register("KBLab/sentence-bert-swedish-cased")
def create_sentence_swedish_cased() -> EmbeddingModel:
    hf_name = "KBLab/sentence-bert-swedish-cased"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["sv"],
        open_source=True,
        embedding_size=768,
    )
    return EmbeddingModel(
        loader=partial(get_sentence_transformer, model_name=hf_name),  # type: ignore
        meta=meta,
    )


@models.register("jonfd/electra-small-nordic")
def create_electra_small_nordic() -> EmbeddingModel:
    hf_name = "jonfd/electra-small-nordic"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["da", "nb", "sv", "nn"],
        open_source=True,
        embedding_size=256,
    )
    return EmbeddingModel(
        loader=partial(get_sentence_transformer, model_name=hf_name),  # type: ignore
        meta=meta,
    )


@models.register("vesteinn/DanskBERT")
def create_dansk_bert() -> EmbeddingModel:
    hf_name = "vesteinn/DanskBERT"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["da"],
        open_source=True,
        embedding_size=768,
    )

    return EmbeddingModel(
        # see https://huggingface.co/vesteinn/DanskBERT/discussions/3
        loader=partial(get_sentence_transformer, model_name=hf_name, max_seq_length=512),  # type: ignore
        meta=meta,
    )


@models.register("chcaa/dfm-encoder-large-v1")
def create_dfm_encoder_large_v1() -> EmbeddingModel:
    hf_name = "chcaa/dfm-encoder-large-v1"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["da"],
        open_source=True,
        embedding_size=1024,
    )
    return EmbeddingModel(
        loader=partial(get_sentence_transformer, model_name=hf_name),  # type: ignore
        meta=meta,
    )


@models.register("NbAiLab/nb-bert-large")
def create_nb_bert_large() -> EmbeddingModel:
    hf_name = "NbAiLab/nb-bert-large"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["nb", "nn"],
        open_source=True,
        embedding_size=1024,
    )
    return EmbeddingModel(
        loader=partial(get_sentence_transformer, model_name=hf_name),  # type: ignore
        meta=meta,
    )


@models.register("NbAiLab/nb-bert-base")
def create_nb_bert_base() -> EmbeddingModel:
    hf_name = "NbAiLab/nb-bert-base"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["nb", "nn"],
        open_source=True,
        embedding_size=768,
    )
    return EmbeddingModel(
        loader=partial(get_sentence_transformer, model_name=hf_name),  # type: ignore
        meta=meta,
    )


"""
# excluded due to loading issues: https://huggingface.co/ltg/norbert3-base/discussions/1
# I could create a custom encoder for this model

@models.register("ltg/norbert3-large")
def create_norbert3_large() -> SebModel:
    hf_name = "ltg/norbert3-large"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["nb", "nn"],
    )
    return SebModel(
        loader=partial(get_sentence_transformer, model_name=hf_name),  # type: ignore
        meta=meta,
    )


@models.register("ltg/norbert3-base")
def create_norbert3_base() -> SebModel:
    hf_name = "ltg/norbert3-base"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["nb", "nn"],
    )
    return SebModel(
        loader=partial(get_sentence_transformer, model_name=hf_name),  # type: ignore
        meta=meta,
    )
"""


@models.register("KB/bert-base-swedish-cased")
def create_bert_base_swedish_cased() -> EmbeddingModel:
    hf_name = "KB/bert-base-swedish-cased"

    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["sv"],
        open_source=True,
        embedding_size=768,
    )
    return EmbeddingModel(
        loader=partial(get_sentence_transformer, model_name=hf_name, max_seq_length=512),  # type: ignore
        meta=meta,
    )


@models.register("kb/electra-small-swedish-cased-discriminator")
def create_electra_small_swedish_cased_discriminator() -> EmbeddingModel:
    hf_name = "kb/electra-small-swedish-cased-discriminator"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["sv"],
        open_source=True,
        embedding_size=256,
    )
    return EmbeddingModel(
        loader=partial(get_sentence_transformer, model_name=hf_name),  # type: ignore
        meta=meta,
    )


# Multilingual baselines
@models.register("xlm-roberta-base")
def create_xlm_roberta_base() -> EmbeddingModel:
    hf_name = "xlm-roberta-base"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        open_source=True,
        embedding_size=768,
    )

    return EmbeddingModel(
        loader=partial(get_sentence_transformer, model_name=hf_name, max_seq_length=512),  # type: ignore
        meta=meta,
    )


# Scandinavian sentence encoders
@models.register("KennethEnevoldsen/dfm-sentence-encoder-large-1")
def create_dfm_sentence_encoder_large() -> EmbeddingModel:
    hf_name = "KennethEnevoldsen/dfm-sentence-encoder-large-1"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["da"],
        open_source=True,
        embedding_size=1024,
    )
    return EmbeddingModel(
        loader=partial(get_sentence_transformer, model_name=hf_name),  # type: ignore
        meta=meta,
    )


@models.register("KennethEnevoldsen/dfm-sentence-encoder-large-exp1")
def create_dfm_sentence_encoder_large_exp() -> EmbeddingModel:
    hf_name = "KennethEnevoldsen/dfm-sentence-encoder-large-exp1"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["da"],
        open_source=True,
        embedding_size=1024,
    )
    return EmbeddingModel(
        loader=partial(get_sentence_transformer, model_name=hf_name),  # type: ignore
        meta=meta,
    )


@models.register("KennethEnevoldsen/dfm-sentence-encoder-small-v1")
def create_dfm_sentence_encoder_small() -> EmbeddingModel:
    hf_name = "KennethEnevoldsen/dfm-sentence-encoder-small-v1"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["da"],
        open_source=True,
        embedding_size=256,
    )
    return EmbeddingModel(
        loader=partial(get_sentence_transformer, model_name=hf_name),  # type: ignore
        meta=meta,
    )


@models.register("KennethEnevoldsen/dfm-sentence-encoder-medium-v1")
def create_dfm_sentence_encoder_medium() -> EmbeddingModel:
    hf_name = "KennethEnevoldsen/dfm-sentence-encoder-medium-v1"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["da"],
        open_source=True,
        embedding_size=768,
    )
    return EmbeddingModel(
        loader=partial(get_sentence_transformer, model_name=hf_name),  # type: ignore
        meta=meta,
    )


@models.register("KennethEnevoldsen/dfm-sentence-encoder-large-exp2-no-lang-align")
def create_dfm_sentence_encoder_large_exp2() -> EmbeddingModel:
    hf_name = "KennethEnevoldsen/dfm-sentence-encoder-large-exp2-no-lang-align"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["da"],
        open_source=True,
        embedding_size=1024,
    )
    return EmbeddingModel(
        loader=partial(get_sentence_transformer, model_name=hf_name),  # type: ignore
        meta=meta,
    )
