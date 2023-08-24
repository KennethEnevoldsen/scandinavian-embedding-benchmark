from collections.abc import Sequence
from pathlib import Path
from typing import Union

import torch
from fairseq2.data import Collater
from fairseq2.data.data_pipeline import read_sequence
from fairseq2.data.text import TextTokenizer, read_text
from fairseq2.models.sequence import SequenceBatch
from fairseq2.typing import Device
from sonar.inference_pipelines.utils import extract_sequence_batch
from sonar.models import SonarEncoderModel, SonarEncoderOutput
from sonar.models.sonar_text import (
    load_sonar_text_encoder_model,
    load_sonar_tokenizer,
)

from seb.model_interface import ModelInterface


def truncate_seq_length(
    sequence_batch: SequenceBatch, max_seq_len: int = 514,
) -> SequenceBatch:
    sequence_batch.seqs = sequence_batch.seqs[:, :max_seq_len]
    sequence_batch.seq_lens = torch.clamp(sequence_batch.seq_lens, max=max_seq_len)  # type: ignore
    return sequence_batch


class SonarTextToEmbeddingModelPipeline(torch.nn.Module, ModelInterface):
    def __init__(
        self,
        encoder: Union[str, SonarEncoderModel],
        tokenizer: Union[str, TextTokenizer],
        device: Device = torch.device("cpu"),
    ) -> None:
        """
        Args:
            encoder (Union[str, SonarEncoderModel]): either cart name or model object
            tokenizer (Union[str, TextTokenizer]): either cart name or tokenizer object
            device (device, optional): . Defaults to cpu.
        """
        super().__init__()
        if isinstance(encoder, str):
            encoder = load_sonar_text_encoder_model(
                encoder, device=device, progress=False,
            )
        if isinstance(tokenizer, str):
            tokenizer = load_sonar_tokenizer(tokenizer, progress=False)

        self.tokenizer = tokenizer
        self.model = encoder.to(device).eval()
        self.device = device

    @torch.inference_mode()
    def encode(
        self, input: Union[Path, Sequence[str]], source_lang: str, batch_size: int,
    ) -> torch.Tensor:
        """Set source_lang to '[dan|swe|nno|nob|]_Latn' for Danish, Swedish,
        Norwegian Nynorsk, and Norwegian Bokmål, respectively."""
        tokenizer_encoder = self.tokenizer.create_encoder(lang=source_lang)

        pipeline = (
            (
                read_text(input)
                if isinstance(input, (str, Path))
                else read_sequence(input)
            )
            .map(tokenizer_encoder)
            .bucket(batch_size)
            .map(Collater(self.tokenizer.vocab_info.pad_idx))
            .map(lambda x: extract_sequence_batch(x, self.device))
            .map(lambda x: truncate_seq_length(x, max_seq_len=514))
            .map(self.model)
            .and_return()
        )

        results: list[SonarEncoderOutput] = list(iter(pipeline))

        sentence_embeddings = torch.cat([x.sentence_embeddings for x in results], dim=0)
        return sentence_embeddings


def get_sonar_model() -> SonarTextToEmbeddingModelPipeline:
    return SonarTextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder",
    )


if __name__ == "__main__":
    pipe = get_sonar_model()

    sents = ["meget lang sætning" * 512] + ["hej"]

    x = pipe.encode(sents, source_lang="dan_Latn", batch_size=32)
