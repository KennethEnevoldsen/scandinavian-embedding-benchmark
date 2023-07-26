"""
All the models registered in the benchmark, along with their metadata.
"""

from functools import partial

from .model_interface import ModelMeta, SebModel
from .registries import models


def get_sentence_transformer(model_name):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


# Relevant multilingual models
@models.register("all-MiniLM-L6-v2")
def create_all_mini_lm_l6_v2() -> SebModel:
    hf_name = "sentence-transformers/all-MiniLM-L6-v2"
    meta = ModelMeta(
        model_name=hf_name.split("/")[-1],
        huggingface_name=f"hf_name",
        model_reference="https://huggingface.co/{hf_name}",
        languages=[],
    )
    return SebModel(
        model_loader=partial(get_sentence_transformer, model_name=hf_name),
        model_meta=meta,
    )


# @models.register("sentence-bert-swedish-cased")
# def create_sentence_swedish_cased() -> SebModel:
#     from sentence_transformers import SentenceTransformer

#     hf_name = "KBLab/sentence-bert-swedish-cased"
#     mdl = SentenceTransformer(hf_name)

#     return SebModel(
#         model_name=hf_name.split("/")[-1],
#         huggingface_name=f"hf_name",
#         model_reference="https://huggingface.co/{hf_name}",
#         languages=["sv"],
#         model=mdl,  # type: ignore
#     )


# @models.register("electra-small-nordic")
# def create_electra_small_nordic() -> SebModel:
#     from sentence_transformers import SentenceTransformer

#     hf_name = "jonfd/electra-small-nordic"
#     mdl = SentenceTransformer(hf_name)

#     return SebModel(
#         model_name=hf_name.split("/")[-1],
#         huggingface_name=f"hf_name",
#         model_reference="https://huggingface.co/{hf_name}",
#         languages=["sv", "da", "no"],
#         model=mdl,  # type: ignore
#     )


# @models.register("DanskBERT")
# def create_dansk_bert() -> SebModel:
#     from sentence_transformers import SentenceTransformer

#     hf_name = "vesteinn/DanskBERT"
#     mdl = SentenceTransformer(hf_name)

#     return SebModel(
#         model_name=hf_name.split("/")[-1],
#         huggingface_name=f"hf_name",
#         model_reference="https://huggingface.co/{hf_name}",
#         languages=["da"],
#         model=mdl,  # type: ignore
#     )


# @models.register("dfm-encoder-large-v1")
# def create_dfm_encoder_large_v1() -> SebModel:
#     from sentence_transformers import SentenceTransformer

#     hf_name = "chcaa/dfm-encoder-large-v1"
#     mdl = SentenceTransformer(hf_name)

#     return SebModel(
#         model_name=hf_name.split("/")[-1],
#         huggingface_name=f"hf_name",
#         model_reference="https://huggingface.co/{hf_name}",
#         languages=["da"],
#         model=mdl,  # type: ignore
#     )


# @models.register("nb-bert-large")
# def create_nb_bert_large() -> SebModel:
#     from sentence_transformers import SentenceTransformer

#     hf_name = "NbAiLab/nb-bert-large"
#     mdl = SentenceTransformer(hf_name)

#     return SebModel(
#         model_name=hf_name.split("/")[-1],
#         huggingface_name=f"hf_name",
#         model_reference="https://huggingface.co/{hf_name}",
#         languages=["nb", "no", "nn"],
#         model=mdl,  # type: ignore
#     )


# @models.register("norbert3-large")
# def create_norbert3_large() -> SebModel:
#     from sentence_transformers import SentenceTransformer

#     hf_name = "ltg/norbert3-large"
#     mdl = SentenceTransformer(hf_name)

#     return SebModel(
#         model_name=hf_name.split("/")[-1],
#         huggingface_name=f"hf_name",
#         model_reference="https://huggingface.co/{hf_name}",
#         languages=["nb", "no", "nn"],
#         model=mdl,  # type: ignore
#     )


# @models.register("norbert3-base")
# def create_norbert3_base() -> SebModel:
#     from sentence_transformers import SentenceTransformer

#     hf_name = "ltg/norbert3-base"
#     mdl = SentenceTransformer(hf_name)

#     return SebModel(
#         model_name=hf_name.split("/")[-1],
#         huggingface_name=f"hf_name",
#         model_reference="https://huggingface.co/{hf_name}",
#         languages=["nb", "no", "nn"],
#         model=mdl,  # type: ignore
#     )


# @models.register("bert-base-swedish-cased")
# def create_bert_base_swedish_cased() -> SebModel:
#     from sentence_transformers import SentenceTransformer

#     hf_name = "KB/bert-base-swedish-cased"
#     mdl = SentenceTransformer(hf_name)

#     return SebModel(
#         model_name=hf_name.split("/")[-1],
#         huggingface_name=f"hf_name",
#         model_reference="https://huggingface.co/{hf_name}",
#         languages=["sv"],
#         model=mdl,  # type: ignore
#     )


# @models.register("electra-small-swedish-cased-discriminator")
# def create_electra_small_swedish_cased_discriminator() -> SebModel:
#     from sentence_transformers import SentenceTransformer

#     hf_name = "KBLab/electra-small-swedish-cased-discriminator"
#     mdl = SentenceTransformer(hf_name)

#     return SebModel(
#         model_name=hf_name.split("/")[-1],
#         huggingface_name=f"hf_name",
#         model_reference="https://huggingface.co/{hf_name}",
#         languages=["sv"],
#         model=mdl,  # type: ignore
#     )


# # Multilingual baselines
# @models.register("xlm-roberta-base")
# def create_xlm_roberta_base() -> SebModel:
#     from sentence_transformers import SentenceTransformer

#     hf_name = "xlm-roberta-base"
#     mdl = SentenceTransformer(hf_name)

#     return SebModel(
#         model_name=hf_name,
#         huggingface_name=f"hf_name",
#         model_reference="https://huggingface.co/{hf_name}",
#         languages=[],
#         model=mdl,  # type: ignore
#     )


# # English
# @models.register("e5-small")
# def create_e5_small() -> SebModel:
#     from sentence_transformers import SentenceTransformer

#     hf_name = "intfloat/e5-small"
#     mdl = SentenceTransformer(hf_name)

#     return SebModel(
#         model_name=hf_name.split("/")[-1],
#         huggingface_name=hf_name,
#         model_reference="https://huggingface.co/{hf_name}",
#         languages=["en"],
#         model=mdl,  # type: ignore
#     )


# @models.register("e5-base")
# def create_e5_base() -> SebModel:
#     from sentence_transformers import SentenceTransformer

#     hf_name = "intfloat/e5-base"
#     mdl = SentenceTransformer(hf_name)

#     return SebModel(
#         model_name=hf_name.split("/")[-1],
#         huggingface_name=hf_name,
#         model_reference="https://huggingface.co/{hf_name}",
#         languages=["en"],
#         model=mdl,  # type: ignore
#     )


# @models.register("e5-large")
# def create_e5_large() -> SebModel:
#     from sentence_transformers import SentenceTransformer

#     hf_name = "intfloat/e5-large"
#     mdl = SentenceTransformer(hf_name)

#     return SebModel(
#         model_name=hf_name.split("/")[-1],
#         huggingface_name=hf_name,
#         model_reference="https://huggingface.co/{hf_name}",
#         languages=["en"],
#         model=mdl,  # type: ignore
#     )


# # Multilingual
# @models.register("multilingual-e5-small")
# def create_multilingual_e5_small() -> SebModel:
#     from sentence_transformers import SentenceTransformer

#     hf_name = "intfloat/multilingual-e5-small"
#     mdl = SentenceTransformer(hf_name)

#     return SebModel(
#         model_name=hf_name.split("/")[-1],
#         huggingface_name=hf_name,
#         model_reference="https://huggingface.co/{hf_name}",
#         languages=[],
#         model=mdl,  # type: ignore
#     )


# @models.register("multilingual-e5-base")
# def create_multilingual_e5_base() -> SebModel:
#     from sentence_transformers import SentenceTransformer

#     hf_name = "intfloat/multilingual-e5-base"
#     mdl = SentenceTransformer(hf_name)

#     return SebModel(
#         model_name=hf_name.split("/")[-1],
#         huggingface_name=hf_name,
#         model_reference="https://huggingface.co/{hf_name}",
#         languages=[],
#         model=mdl,  # type: ignore
#     )


# @models.register("multilingual-e5-large")
# def create_multilingual_e5_large() -> SebModel:
#     from sentence_transformers import SentenceTransformer

#     hf_name = "intfloat/multilingual-e5-large"
#     mdl = SentenceTransformer(hf_name)

#     return SebModel(
#         model_name=hf_name.split("/")[-1],
#         huggingface_name=hf_name,
#         model_reference="https://huggingface.co/{hf_name}",
#         languages=[],
#         model=mdl,  # type: ignore
#     )


# # my model
# @models.register("dfm-sentence-encoder-large-1")
# def create_dfm_sentence_encoder_large_1() -> SebModel:
#     from sentence_transformers import SentenceTransformer

#     hf_name = "KennethEnevoldsen/dfm-sentence-encoder-large-1"
#     mdl = SentenceTransformer(hf_name)

#     return SebModel(
#         model_name=hf_name.split("/")[-1],
#         huggingface_name=hf_name,
#         model_reference="https://huggingface.co/{hf_name}",
#         languages=["da"],
#         model=mdl,  # type: ignore
#     )
