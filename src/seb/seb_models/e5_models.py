from functools import partial

from seb import EmbeddingModel, ModelInterface, ModelMeta, models
from seb.model_interface import ArrayLike

from .hf_models import get_sentence_transformer


class E5Wrapper(ModelInterface):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.mdl = get_sentence_transformer(model_name)

    @staticmethod
    def preprocess(sentences: list[str]) -> list[str]:
        # following the documentation it is better to generally do this:
        return ["query: " + sentence for sentence in sentences]

    # but it does not work slightly better than this:
    # return sentences # noqa

    def encode(
        self,
        sentences: list[str],
        batch_size: int = 32,
        **kwargs: dict,
    ) -> list[ArrayLike]:
        sentences = self.preprocess(sentences)
        return self.mdl.encode(sentences, batch_size=batch_size, **kwargs)  # type: ignore


# English
@models.register("intfloat/e5-small")
def create_e5_small() -> EmbeddingModel:
    hf_name = "intfloat/e5-small"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["en"],
        open_source=True,
        embedding_size=384,
    )
    return EmbeddingModel(
        loader=partial(E5Wrapper, model_name=hf_name),  # type: ignore
        meta=meta,
    )


@models.register("intfloat/e5-base")
def create_e5_base() -> EmbeddingModel:
    hf_name = "intfloat/e5-base"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["en"],
        open_source=True,
        embedding_size=768,
    )
    return EmbeddingModel(
        loader=partial(E5Wrapper, model_name=hf_name),  # type: ignore
        meta=meta,
    )


@models.register("intfloat/e5-large")
def create_e5_large() -> EmbeddingModel:
    hf_name = "intfloat/e5-large"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["en"],
        open_source=True,
        embedding_size=1024,
    )
    return EmbeddingModel(
        loader=partial(E5Wrapper, model_name=hf_name),  # type: ignore
        meta=meta,
    )


# Multilingual
@models.register("intfloat/multilingual-e5-small")
def create_multilingual_e5_small() -> EmbeddingModel:
    hf_name = "intfloat/multilingual-e5-small"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=[],
        open_source=True,
        embedding_size=384,
    )
    return EmbeddingModel(
        loader=partial(E5Wrapper, model_name=hf_name),  # type: ignore
        meta=meta,
    )


@models.register("intfloat/multilingual-e5-base")
def create_multilingual_e5_base() -> EmbeddingModel:
    hf_name = "intfloat/multilingual-e5-base"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=[],
        open_source=True,
        embedding_size=768,
    )
    return EmbeddingModel(
        loader=partial(E5Wrapper, model_name=hf_name),  # type: ignore
        meta=meta,
    )


@models.register("intfloat/multilingual-e5-large")
def create_multilingual_e5_large() -> EmbeddingModel:
    hf_name = "intfloat/multilingual-e5-large"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=[],
        open_source=True,
        embedding_size=1024,
    )
    return EmbeddingModel(
        loader=partial(E5Wrapper, model_name=hf_name),  # type: ignore
        meta=meta,
    )


@models.register("intfloat/e5-mistral-7b-instruct")
def create_multilingual_e5_mistral_7b_instruct() -> EmbeddingModel:
    hf_name = "intfloat/e5-mistral-7b-instruct"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=[],
        open_source=True,
        embedding_size=4096,
    )
    return EmbeddingModel(
        loader=partial(E5Wrapper, model_name=hf_name),  # type: ignore
        meta=meta,
    )
