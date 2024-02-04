from collections.abc import Iterable, Sequence
from itertools import islice
from typing import Any, Optional, TypeVar

import torch
import torch.nn.functional as F
from numpy.typing import ArrayLike
from torch import Tensor
from transformers import AutoModel, AutoTokenizer, BatchEncoding

from seb.interfaces.model import Encoder, LazyLoadEncoder, ModelMeta, SebModel
from seb.interfaces.task import Task
from seb.registries import models

T = TypeVar("T")


def batched(iterable: Iterable[T], n: int) -> Iterable[tuple[T, ...]]:
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


class E5Mistral(Encoder):
    max_length = 4096

    def __init__(self):
        self.load_model()

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-mistral-7b-instruct")
        self.model = AutoModel.from_pretrained("intfloat/e5-mistral-7b-instruct")

    def preprocess(self, sentences: Sequence[str]) -> BatchEncoding:
        # following the documentation we should also add "Instruction: " to the instruction, but for now I will just create this naive approach
        # task = ""  # Could e.g. be: "Given a web search query, retrieve relevant passages that answer the query"  # noqa
        # And then:
        # f"Instruction: {task} Query: {sentence}" for sentence in sentences
        sentences = ["Query: " + sentence for sentence in sentences]
        batch_dict = self.tokenizer(
            sentences,
            max_length=self.max_length - 1,
            return_attention_mask=False,
            padding=False,
            truncation=True,
        )
        # append eos_token_id to every input_ids
        batch_dict["input_ids"] = [
            [*input_ids, self.tokenizer.eos_token_id]
            for input_ids in batch_dict["input_ids"]  # type: ignore
        ]
        batch_dict = self.tokenizer.pad(batch_dict, padding=True, return_attention_mask=True, return_tensors="pt")

        return batch_dict

    # but it does not work slightly better than this:
    # return sentences # noqa

    @staticmethod
    def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]

    def encode(
        self,
        sentences: list[str],
        *,
        task: Optional[Task] = None,  # noqa
        batch_size: int = 8,
        **kwargs: Any,  # noqa
    ) -> ArrayLike:
        batched_embeddings = []
        for batch in batched(sentences, batch_size):
            batch_dict = self.preprocess(batch)

            outputs = self.model(**batch_dict)
            embeddings = self.last_token_pool(
                outputs.last_hidden_state,
                batch_dict["attention_mask"],  # type: ignore
            )

            # normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            scores = (embeddings[:2] @ embeddings[2:].T) * 100
            batched_embeddings.append(scores)

        return torch.cat(batched_embeddings)


@models.register("intfloat/e5-mistral-7b-instruct")
def create_multilingual_e5_mistral_7b_instruct() -> SebModel:
    hf_name = "intfloat/e5-mistral-7b-instruct"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=[],
        open_source=True,
        embedding_size=4096,
    )
    return SebModel(
        encoder=LazyLoadEncoder(E5Mistral),
        meta=meta,
    )
