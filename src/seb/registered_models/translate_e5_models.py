from collections.abc import Sequence
from functools import partial
from typing import Any, Optional

import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

import seb
from seb import models
from seb.registered_models.e5_models import E5Wrapper
from seb.registries import models


class TranslateE5Model(seb.Encoder):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.mdl = E5Wrapper(model_name)
        self.trans_model: M2M100ForConditionalGeneration = (  # type: ignore
            M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
        )
        self.trans_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

    def translate(self, sentence: str, src_lang: str) -> str:
        self.trans_tokenizer.src_lang = src_lang
        encoded_sent = self.trans_tokenizer(sentence, return_tensors="pt")
        gen_tokens = self.trans_model.generate(**encoded_sent, forced_bos_token_id=self.trans_tokenizer.get_lang_id("en"))
        return self.trans_tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]

    def encode(
        self,
        sentences: list[str],
        *,
        task: Optional[seb.Task] = None,  # noqa: ARG002
        batch_size: int = 32,
        **kwargs: Any,
    ) -> torch.Tensor:
        try:
            src_lang = task.languages[0]  # type: ignore
        except IndexError:
            # Danish is the default fallback if no language is specified for the task.
            src_lang = "da"
        sentences = [self.translate(sentence, src_lang) for sentence in sentences]
        return self.mdl.encode(sentences, task=task, batch_size=batch_size, **kwargs)  # type: ignore


@models.register("translate-e5-small")
def create_translate_e5_small() -> seb.EmbeddingModel:
    hf_name = "intfloat/e5-small"
    meta = seb.ModelMeta(
        name="translate-e5-small",
        reference=f"https://huggingface.co/{hf_name}",
        languages=["en"],
        open_source=True,
        embedding_size=384,
    )
    return seb.EmbeddingModel(
        loader=partial(TranslateE5Model, model_name=hf_name),  # type: ignore
        meta=meta,
    )


@models.register("translate-e5-base")
def create_translate_e5_base() -> seb.EmbeddingModel:
    hf_name = "intfloat/e5-base"
    meta = seb.ModelMeta(
        name="translate-e5-base",
        reference=f"https://huggingface.co/{hf_name}",
        languages=["en"],
        open_source=True,
        embedding_size=384,
    )
    return seb.EmbeddingModel(
        loader=partial(TranslateE5Model, model_name=hf_name),  # type: ignore
        meta=meta,
    )


@models.register("translate-e5-large")
def create_translate_e5_large() -> seb.EmbeddingModel:
    hf_name = "intfloat/e5-large"
    meta = seb.ModelMeta(
        name="translate-e5-large",
        reference=f"https://huggingface.co/{hf_name}",
        languages=["en"],
        open_source=True,
        embedding_size=384,
    )
    return seb.EmbeddingModel(
        loader=partial(TranslateE5Model, model_name=hf_name),  # type: ignore
        meta=meta,
    )
