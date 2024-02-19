import logging
from collections.abc import Iterable, Sequence
from datetime import date
from itertools import islice
from typing import Any, Literal, Optional, TypeVar

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BatchEncoding

from seb.interfaces.model import Encoder, LazyLoadEncoder, ModelMeta, SebModel
from seb.interfaces.task import Task
from seb.registries import models

logger = logging.getLogger(__name__)


T = TypeVar("T")
EncodeTypes = Literal["query", "passage"]


def batched(iterable: Iterable[T], n: int) -> Iterable[tuple[T, ...]]:
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def batch_to_device(batch_data: dict[str, torch.Tensor], device: str = "cuda") -> dict[str, torch.Tensor]:
    return {key: data.to(device) for key, data in batch_data.items()}


def task_to_instruction(task: Task) -> str:
    if task.task_type in ["STS"]:
        return "Retrieve semantically similar text"
    if task.task_type in ["Summarization"]:
        return "Given a news summary, retrieve other semantically similar summaries"
    if task.task_type in ["BitextMining"]:
        task_name_to_instruct: dict[str, str] = {
            "Bornholm Parallel": "Retrieve parallel sentences in Danish and Bornholmsk",
            "Norwegian courts": "Retrieve parallel sentences in Norwegian Bokmål and Nynorsk",
        }
        default_instruction = "Retrieve parallel sentences."
        return task_name_to_instruct.get(task.name, default_instruction)
    if task.task_type in ["Classification"]:
        task_name_to_instruct: dict[str, str] = {
            "Angry Tweets": "Classify Danish tweets by sentiment. (positive, negative, neutral)",
            "DKHate": "Classify Danish tweets based on offensiveness (offensive, not offensive)",
            "Da Political Comments": "Classify Danish political comments for sentiment",
            "DaLAJ": "Classify texts based on linguistic acceptability in Swedish",
            "LCC": "Classify texts based on sentiment",
            "Language Identification": "Classify texts based on language",
            "Massive Intent": "Given a user utterance as query, find the user intents",
            "Massive Scenario": "Given a user utterance as query, find the user scenarios",
            "NoReC": "Classify Norwegian reviews by sentiment",
            "SweReC": "Classify Swedish reviews by sentiment",
            "Norwegian parliament": "Classify parliament speeches in Norwegian based on political affiliation",
            "ScaLA": "Classify passages in Scandinavian Languages based on linguistic acceptability",
        }
        default_instruction = "Classify user passages"
        return task_name_to_instruct.get(task.name, default_instruction)
    if task.task_type in ["Clustering"]:
        task_name_to_instruct: dict[str, str] = {
            "ArxivClusteringP2P": "Identify the main and secondary category of Arxiv papers based on the titles and abstracts",
            "VG Clustering": "Identify the categories (e.g. sports) of given articles in Norwegian",
            "SNL Clustering": "Identify categories in a Norwegian lexicon",
            "SwednClustering": "Identify news categories in Swedish passages",
        }
        default_instruction = "Identify categories in user passages"
        return task_name_to_instruct.get(task.name, default_instruction)
    if task.task_type in ["Reranking"]:
        return "Retrieve semantically similar passages."
    if task.task_type in ["Retrieval"]:
        task_name_to_instruct: dict[str, str] = {
            "Twitterhjerne": "Retrieve answers to questions asked in Danish tweets",
            "SwednRetrieval": "Retrieve summaries of Swedish news articles",
            "TV2Nord Retrieval": "Retrieve summaries of Danish news articles",
            "DanFEVER": "Given a claim in Danish, retrieve documents that support or refute the claim",
            "SNL Retrieval": "Given a lexicon article in Norwegian, retrieve its headline",
            "NorQuad": "Given a question in Norwegian, retrieve the answer from Wikipedia articles",
            "SweFAQ": "Retrieve answers given questions in Swedish",
            "ArguAna": "Given a claim, find documents that refute the claim",
            "ClimateFEVER": "Given a claim about climate change, retrieve documents that support or refute the claim",
        }
        default_instruction = "Retrieve text based on user query."
        return task_name_to_instruct.get(task.name, default_instruction)
    return ""


class E5Mistral(Encoder):
    max_length = 4096
    max_batch_size = 4

    def __init__(self):
        logger.info("Started loading e5 Mistral")
        self.tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-mistral-7b-instruct")
        self.model = AutoModel.from_pretrained("intfloat/e5-mistral-7b-instruct", torch_dtype=torch.float16)

    def preprocess(self, sentences: Sequence[str], instruction: str, encode_type: EncodeTypes) -> BatchEncoding:
        if encode_type == "query":
            sentences = [f"Instruction: {instruction}\nQuery: {sentence}" for sentence in sentences]
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

        return batch_dict.to(self.model.device)

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
        task: Optional[Task] = None,
        batch_size: int = 32,
        encode_type: EncodeTypes = "query",
        **kwargs: Any,  # noqa
    ) -> np.ndarray:
        if batch_size > self.max_batch_size:
            batch_size = self.max_batch_size
        batched_embeddings = []
        if task is not None:  # noqa
            instruction = task_to_instruction(task)
        else:
            instruction = ""
        for batch in tqdm(batched(sentences, batch_size)):
            with torch.inference_mode():
                batch_dict = self.preprocess(batch, instruction=instruction, encode_type=encode_type)
                outputs = self.model(**batch_dict)
                embeddings = self.last_token_pool(
                    outputs.last_hidden_state,
                    batch_dict["attention_mask"],  # type: ignore
                )
            batched_embeddings.append(embeddings.detach().cpu())

        return torch.cat(batched_embeddings).to("cpu").detach().numpy()

    def encode_corpus(self, corpus: list[dict[str, str]], **kwargs: Any) -> np.ndarray:
        sep = " "
        if isinstance(corpus, dict):
            sentences = [
                (corpus["title"][i] + sep + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][i].strip()  # type: ignore
                for i in range(len(corpus["text"]))  # type: ignore
            ]
        else:
            sentences = [(doc["title"] + sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus]
        return self.encode(sentences, encode_type="passage", **kwargs)

    def encode_queries(self, queries: list[str], **kwargs: Any) -> np.ndarray:
        return self.encode(queries, encode_type="query", **kwargs)


@models.register("e5-mistral-7b-instruct")
def create_multilingual_e5_mistral_7b_instruct() -> SebModel:
    hf_name = "intfloat/e5-mistral-7b-instruct"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=[],
        open_source=True,
        embedding_size=4096,
        model_architecture="Mistral",
        release_date=date(2023, 12, 20),
    )
    return SebModel(
        encoder=LazyLoadEncoder(E5Mistral),
        meta=meta,
    )
