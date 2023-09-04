from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Union

import torch

from seb.model_interface import ModelInterface, ModelMeta, SebModel
from seb.registries import models


def truncate_seq_length(
    sequence_batch,  # SequenceBatch ,
    max_seq_len: int = 514,
):  # -> SequenceBatch:
    sequence_batch.seqs = sequence_batch.seqs[:, :max_seq_len]
    sequence_batch.seq_lens = torch.clamp(sequence_batch.seq_lens, max=max_seq_len)  # type: ignore
    return sequence_batch


class SonarTextToEmbeddingModelPipeline(torch.nn.Module, ModelInterface):
    def __init__(
        self,
        encoder_name: str,
        tokenizer_name: str,
        source_lang: str,
        device: torch.device = torch.device("cpu"),
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
    def encode(
        self,
        input: Union[Path, Sequence[str]],
        batch_size: int,
    ) -> torch.Tensor:
        from fairseq2.data import Collater  # type: ignore
        from fairseq2.data.data_pipeline import read_sequence  # type: ignore
        from fairseq2.data.text import read_text  # type: ignore

        from sonar.inference_pipelines.utils import extract_sequence_batch  # type: ignore # isort: skip

        tokenizer_encoder = self.tokenizer.create_encoder(lang=self.source_lang)  # type: ignore

        pipeline = (
            (
                read_text(input)
                if isinstance(input, (str, Path))
                else read_sequence(input)
            )
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
        msg = (
            "Could not fetch Sonar Models. Make sure you have"
            + "fairseq2 installed. This is currently only supported for "
            + "Linux."
        )
        raise ImportError(msg)


@models.register("facebook/SONAR_da")
def create_sonar_da() -> SebModel:
    meta = ModelMeta(
        name="sonar_dan",
        huggingface_name=None,
        reference="https://github.com/facebookresearch/SONAR",
        languages=["da"],
    )
    return SebModel(
        loader=partial(get_sonar_model, source_lang="dan_Latn"),
        meta=meta,
    )


@models.register("facebook/SONAR_sv")
def create_sonar_sv() -> SebModel:
    meta = ModelMeta(
        name="sonar_swe",
        huggingface_name=None,
        reference="https://github.com/facebookresearch/SONAR",
        languages=["sv"],
    )
    return SebModel(
        loader=partial(get_sonar_model, source_lang="swe_Latn"),
        meta=meta,
    )


@models.register("facebook/SONAR_nb")
def create_sonar_nb() -> SebModel:
    meta = ModelMeta(
        name="sonar_nob",
        huggingface_name=None,
        reference="https://github.com/facebookresearch/SONAR",
        languages=["nb"],
    )
    return SebModel(
        loader=partial(get_sonar_model, source_lang="nob_Latn"),
        meta=meta,
    )


@models.register("facebook/SONAR_nn")
def create_sonar_nn() -> SebModel:
    meta = ModelMeta(
        name="sonar_nno",
        huggingface_name=None,
        reference="https://github.com/facebookresearch/SONAR",
        languages=["nn"],
    )
    return SebModel(
        loader=partial(get_sonar_model, source_lang="nno_Latn"),
        meta=meta,
    )
