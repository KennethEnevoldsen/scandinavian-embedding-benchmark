from collections.abc import Sequence
from functools import partial
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
from seb.model_interface import ModelMeta, SebModel
from seb.registries import models

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
        source_lang: str,
        device: Device = torch.device("cpu"),
    ) -> None:
        """
        Args:
            encoder (Union[str, SonarEncoderModel]): either cart name or model object
            tokenizer (Union[str, TextTokenizer]): either cart name or tokenizer object
            device (device, optional): . Defaults to cpu
            Set source_lang to '[dan|swe|nno|nob|]_Latn' for Danish, Swedish,
            Norwegian Nynorsk, and Norwegian Bokmål, respectively.
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
        self.source_lang = source_lang

    @torch.inference_mode()
    def encode(
        self, input: Union[Path, Sequence[str]], batch_size: int,
    ) -> torch.Tensor:
    
        tokenizer_encoder = self.tokenizer.create_encoder(lang=self.source_lang)

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
        return sentence_embeddings.numpy()


def get_sonar_model(source_lang: str) -> SonarTextToEmbeddingModelPipeline:
    return SonarTextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder", source_lang=source_lang
    )



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



if __name__ == "__main__":
    from seb.seb_models.hf_models import create_all_mini_lm_l6_v2


    sonar = create_sonar_da()
    st = create_all_mini_lm_l6_v2()

    sents = ["Hej "* 60]

    sonar_out = sonar.encode(sents)
    st_out = st.encode(sents)

    print(sonar_out.shape)
    print(st_out.shape)
    pass