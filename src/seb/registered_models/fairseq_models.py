from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any, Optional, Union

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


class SonarTextToEmbeddingModelPipeline(torch.nn.Module, Encoder):
    def __init__(
        self,
        encoder_name: str,
        tokenizer_name: str,
        source_lang: str,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Args:
            encoder_name: Name of the encoder model
            tokenizer_name: Name of the tokenizer
            device: Defaults to cpu
            source_lang: Set source_lang to '[dan|swe|nno|nob|]_Latn' for Danish, Swedish,
                Norwegian Nynorsk, and Norwegian Bokmål, respectively.
        """
        from sonar.models.sonar_text import (  # type: ignore
            load_sonar_text_encoder_model,
            load_sonar_tokenizer,
        )

        super().__init__()

        if device is None:
            device = torch.device("cpu")

        _encoder = load_sonar_text_encoder_model(
            encoder_name,
            device=device,
            progress=False,
        )
        tokenizer_name = load_sonar_tokenizer(tokenizer_name, progress=False)

        self.tokenizer = tokenizer_name
        self.model = _encoder.to(device).eval()
        self.device = device
        self.source_lang = source_lang

    @torch.inference_mode()
    def encode(  # type: ignore
        self,
        input: Union[Path, Sequence[str]],  # noqa: A002
        *,
        task: Optional[Task] = None,
        batch_size: int,
        **kwargs: Any,  # noqa: ARG002
    ) -> torch.Tensor:
        from fairseq2.data import Collater  # type: ignore
        from fairseq2.data.data_pipeline import read_sequence  # type: ignore
        from fairseq2.data.text import read_text  # type: ignore

        from sonar.inference_pipelines.utils import extract_sequence_batch  # type: ignore # isort: skip

        tokenizer_encoder = self.tokenizer.create_encoder(lang=self.source_lang)  # type: ignore

        pipeline = (
            (read_text(input) if isinstance(input, (str, Path)) else read_sequence(input))
            .map(tokenizer_encoder)
            .bucket(batch_size)
            .map(Collater(self.tokenizer.vocab_info.pad_idx))  # type: ignore
            .map(lambda x: extract_sequence_batch(x, self.device))
            .map(lambda x: truncate_seq_length(x, max_seq_len=514))
            .map(self.model)
            .and_return()
        )

        results = list(iter(pipeline))

        sentence_embeddings = torch.cat([x.sentence_embeddings for x in results], dim=0)
        return sentence_embeddings.numpy()


def get_sonar_model(source_lang: str) -> SonarTextToEmbeddingModelPipeline:
    try:
        return SonarTextToEmbeddingModelPipeline(
            encoder_name="text_sonar_basic_encoder",
            tokenizer_name="text_sonar_basic_encoder",
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
    )
    return SebModel(
        encoder=LazyLoadEncoder(partial(get_sonar_model, source_lang="nno_Latn")),
        meta=meta,
    )
