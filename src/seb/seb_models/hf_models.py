"""
All the models registered in the benchmark, along with their metadata.
"""
import logging
from functools import partial

from sentence_transformers import SentenceTransformer

from seb.model_interface import ModelMeta, SebModel
from seb.registries import models


def silence_warnings_from_sentence_transformers():
    from sentence_transformers.SentenceTransformer import logger

    logger.setLevel(logging.ERROR)


def get_sentence_transformer(
    model_name: str, max_seq_length=None,
) -> SentenceTransformer:
    silence_warnings_from_sentence_transformers()
    mdl = SentenceTransformer(model_name)
    if max_seq_length is not None:
        mdl.max_seq_length = max_seq_length
    return mdl


# Relevant multilingual models
@models.register("sentence-transformers/all-MiniLM-L6-v2")
def create_all_mini_lm_l6_v2() -> SebModel:
    hf_name = "sentence-transformers/all-MiniLM-L6-v2"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["en"],
    )
    return SebModel(
        loader=partial(get_sentence_transformer, model_name=hf_name),  # type: ignore
        meta=meta,
    )


@models.register("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
def create_multilingual_mini_lm_l12_v2() -> SebModel:
    hf_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=[],
    )
    return SebModel(
        loader=partial(get_sentence_transformer, model_name=hf_name),  # type: ignore
        meta=meta,
    )


@models.register("KBLab/sentence-bert-swedish-cased")
def create_sentence_swedish_cased() -> SebModel:
    hf_name = "KBLab/sentence-bert-swedish-cased"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["sv"],
    )
    return SebModel(
        loader=partial(get_sentence_transformer, model_name=hf_name),  # type: ignore
        meta=meta,
    )


@models.register("jonfd/electra-small-nordic")
def create_electra_small_nordic() -> SebModel:
    hf_name = "jonfd/electra-small-nordic"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["da", "nb", "sv", "nn"],
    )
    return SebModel(
        loader=partial(get_sentence_transformer, model_name=hf_name),  # type: ignore
        meta=meta,
    )


@models.register("vesteinn/DanskBERT")
def create_dansk_bert() -> SebModel:
    hf_name = "vesteinn/DanskBERT"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["da"],
    )

    return SebModel(
        # see https://huggingface.co/vesteinn/DanskBERT/discussions/3
        loader=partial(get_sentence_transformer, model_name=hf_name, max_seq_length=512),  # type: ignore
        meta=meta,
    )


@models.register("chcaa/dfm-encoder-large-v1")
def create_dfm_encoder_large_v1() -> SebModel:
    hf_name = "chcaa/dfm-encoder-large-v1"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["da"],
    )
    return SebModel(
        loader=partial(get_sentence_transformer, model_name=hf_name),  # type: ignore
        meta=meta,
    )


@models.register("NbAiLab/nb-bert-large")
def create_nb_bert_large() -> SebModel:
    hf_name = "NbAiLab/nb-bert-large"
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


@models.register("NbAiLab/nb-bert-base")
def create_nb_bert_base() -> SebModel:
    hf_name = "NbAiLab/nb-bert-base"
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


# excluded due to loading issues: https://huggingface.co/ltg/norbert3-base/discussions/1
# I could create a custom encoder for this model

# @models.register("ltg/norbert3-large")
# def create_norbert3_large() -> SebModel:
#     hf_name = "ltg/norbert3-large"
#     meta = ModelMeta(
#         name=hf_name.split("/")[-1],
#         huggingface_name=hf_name,
#         reference=f"https://huggingface.co/{hf_name}",
#         languages=["nb", "nn"],
#     )
#     return SebModel(
#         loader=partial(get_sentence_transformer, model_name=hf_name),  # type: ignore
#         meta=meta,
#     )


# @models.register("ltg/norbert3-base")
# def create_norbert3_base() -> SebModel:
#     hf_name = "ltg/norbert3-base"
#     meta = ModelMeta(
#         name=hf_name.split("/")[-1],
#         huggingface_name=hf_name,
#         reference=f"https://huggingface.co/{hf_name}",
#         languages=["nb", "nn"],
#     )
#     return SebModel(
#         loader=partial(get_sentence_transformer, model_name=hf_name),  # type: ignore
#         meta=meta,
#     )


@models.register("KB/bert-base-swedish-cased")
def create_bert_base_swedish_cased() -> SebModel:
    hf_name = "KB/bert-base-swedish-cased"

    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["sv"],
    )
    return SebModel(
        loader=partial(get_sentence_transformer, model_name=hf_name, max_seq_length=512),  # type: ignore
        meta=meta,
    )


@models.register("kb/electra-small-swedish-cased-discriminator")
def create_electra_small_swedish_cased_discriminator() -> SebModel:
    hf_name = "kb/electra-small-swedish-cased-discriminator"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["sv"],
    )
    return SebModel(
        loader=partial(get_sentence_transformer, model_name=hf_name),  # type: ignore
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
    )

    return SebModel(
        loader=partial(get_sentence_transformer, model_name=hf_name, max_seq_length=512),  # type: ignore
        meta=meta,
    )


# English
@models.register("intfloat/e5-small")
def create_e5_small() -> SebModel:
    hf_name = "intfloat/e5-small"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["en"],
    )
    return SebModel(
        loader=partial(get_sentence_transformer, model_name=hf_name),  # type: ignore
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
    )
    return SebModel(
        loader=partial(get_sentence_transformer, model_name=hf_name),  # type: ignore
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
    )
    return SebModel(
        loader=partial(get_sentence_transformer, model_name=hf_name),  # type: ignore
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
    )
    return SebModel(
        loader=partial(get_sentence_transformer, model_name=hf_name),  # type: ignore
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
    )
    return SebModel(
        loader=partial(get_sentence_transformer, model_name=hf_name),  # type: ignore
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
    )
    return SebModel(
        loader=partial(get_sentence_transformer, model_name=hf_name),  # type: ignore
        meta=meta,
    )



# Scandinavian sentence encoders
@models.register("KennethEnevoldsen/dfm-sentence-encoder-large-1")
def create_dfm_sentence_encoder_large() -> SebModel:
    hf_name = "KennethEnevoldsen/dfm-sentence-encoder-large-1"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["da"],
    )
    return SebModel(
        loader=partial(get_sentence_transformer, model_name=hf_name),  # type: ignore
        meta=meta,
    )
