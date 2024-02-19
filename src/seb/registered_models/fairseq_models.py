from collections.abc import Sequence
from datetime import date
from functools import partial
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch

from seb.interfaces.model import Encoder, LazyLoadEncoder, ModelMeta, SebModel
from seb.interfaces.task import Task
from seb.registries import models


def truncate_seq_length(  # noqa: ANN201
    sequence_batch,  # SequenceBatch ,  # noqa: ANN001
    max_seq_len: int = 514,
):  # -> SequenceBatch:
    sequence_batch.seqs = sequence_batch.seqs[:, :max_seq_len]
    sequence_batch.seq_lens = torch.clamp(sequence_batch.seq_lens, max=max_seq_len)  # type: ignore
    return sequence_batch


class SonarTextToEmbeddingModelPipeline(Encoder):
    def __init__(
        self,
        source_lang: str,
    ) -> None:
        """
        Args:
            encoder_name: Name of the encoder model
            tokenizer_name: Name of the tokenizer
            source_lang: Set source_lang to '[dan|swe|nno|nob|]_Latn' for Danish, Swedish,
                Norwegian Nynorsk, and Norwegian Bokmål, respectively.
        """
        from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline  # type: ignore

        self.t2vec_model = TextToEmbeddingModelPipeline(encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder")
        self.source_lang = source_lang

    def to(self, device: torch.device) -> None:
        self.model = self.t2vec_model.to(device)
        self.device = device

    @torch.inference_mode()
    def encode(  # type: ignore
        self,
        sentences: Union[Path, Sequence[str]],
        *,
        task: Optional[Task] = None,  # noqa: ARG002
        batch_size: int,
        **kwargs: Any,  # noqa: ARG002
    ) -> np.ndarray:
        sentence_embeddings = self.t2vec_model.predict(sentences, source_lang=self.source_lang, batch_size=batch_size)
        return sentence_embeddings.numpy()


def get_sonar_model(source_lang: str) -> SonarTextToEmbeddingModelPipeline:
    try:
        return SonarTextToEmbeddingModelPipeline(
            source_lang=source_lang,
        )
    except ImportError:
        msg = "Could not fetch Sonar Models. Make sure you have" + " fairseq2 installed. This is currently only supported for " + "Linux."
        raise ImportError(msg)  # noqa B904


@models.register("sonar-dan")
def create_sonar_da() -> SebModel:
    meta = ModelMeta(
        name="sonar-dan",
        huggingface_name=None,
        reference="https://github.com/facebookresearch/SONAR",
        languages=["da"],
        open_source=True,
        embedding_size=1024,
        architecture="SONAR",
        release_date=date(2023, 5, 17),
    )
    return SebModel(
        encoder=LazyLoadEncoder(partial(get_sonar_model, source_lang="dan_Latn")),
        meta=meta,
    )


@models.register("sonar-swe")
def create_sonar_sv() -> SebModel:
    meta = ModelMeta(
        name="sonar-swe",
        huggingface_name=None,
        reference="https://github.com/facebookresearch/SONAR",
        languages=["sv"],
        open_source=True,
        embedding_size=1024,
        architecture="SONAR",
        release_date=date(2023, 5, 17),
    )
    return SebModel(
        encoder=LazyLoadEncoder(partial(get_sonar_model, source_lang="swe_Latn")),
        meta=meta,
    )


@models.register("sonar-nob")
def create_sonar_nb() -> SebModel:
    meta = ModelMeta(
        name="sonar-nob",
        huggingface_name=None,
        reference="https://github.com/facebookresearch/SONAR",
        languages=["nb"],
        open_source=True,
        embedding_size=1024,
        architecture="SONAR",
        release_date=date(2023, 5, 17),
    )
    return SebModel(
        encoder=LazyLoadEncoder(partial(get_sonar_model, source_lang="nob_Latn")),
        meta=meta,
    )


@models.register("sonar-nno")
def create_sonar_nn() -> SebModel:
    meta = ModelMeta(
        name="sonar-nno",
        huggingface_name=None,
        reference="https://github.com/facebookresearch/SONAR",
        languages=["nn"],
        open_source=True,
        embedding_size=1024,
        release_date=date(2023, 5, 17),
    )
    return SebModel(
        encoder=LazyLoadEncoder(partial(get_sonar_model, source_lang="nno_Latn")),
        meta=meta,
    )
