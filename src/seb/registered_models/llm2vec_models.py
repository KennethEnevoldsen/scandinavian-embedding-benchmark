from __future__ import annotations
import logging
from datetime import date
from functools import partial
import torch
from typing import Any, Optional, TypeVar, Union, List
from collections.abc import Iterable, Sequence
from tqdm import tqdm
from itertools import islice
import numpy as np

import seb
from seb.interfaces.model import LazyLoadEncoder, ModelMeta, SebModel, Encoder
from seb.interfaces.task import Task
from seb.registries import models


logger = logging.getLogger(__name__)
T = TypeVar("T")


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
            "SwednRetrieval": "Given a Swedish news headline retrieve summaries or news articles",
            "TV2Nord Retrieval": "Given a summary of a Danish news article retrieve the corresponding news article",
            "DanFEVER": "Given a claim in Danish, retrieve documents that support the claim",
            "SNL Retrieval": "Given a lexicon headline in Norwegian, retrieve its article",
            "NorQuad": "Given a question in Norwegian, retrieve the answer from Wikipedia articles",
            "SweFAQ": "Retrieve answers given questions in Swedish",
            "ArguAna": "Given a claim, find documents that refute the claim",
            "ClimateFEVER": "Given a claim about climate change, retrieve documents that support or refute the claim",
        }
        default_instruction = "Retrieve text based on user query."
        return task_name_to_instruct.get(task.name, default_instruction)
    return ""


def batched(iterable: Iterable[T], n: int) -> Iterable[tuple[T, ...]]:
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


class LLM2VecModel(Encoder):
    def __init__(
        self,
        base_model_name_or_path: str,
        peft_model_name_or_path: str,
        max_length: int,
        max_batch_size: Optional[int] = None,
        **kwargs: Any,
    ):
        logger.info("Started loading LLM2Vec model")
        try:
            from llm2vec import LLM2Vec
        except ImportError:
            raise ImportError("To use the LLM2Vec models `llm2vec` is required. Please install it with `pip seb[llm2vec].")
        extra_kwargs = {}
        try:
            import flash_attn  # noqa

            extra_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            logger.warning(
                "LLM2Vec models were trained with flash attention enabled. For optimal performance, please install the `flash_attn` package"
            )
        self.model = LLM2Vec.from_pretrained(
            base_model_name_or_path=base_model_name_or_path,
            peft_model_name_or_path=peft_model_name_or_path,
            **extra_kwargs,
            **kwargs,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.max_length = max_length
        self.max_batch_size = max_batch_size

    def preprocess(self, sentences, instruction) -> list[str] | list[list[str]]:
        if instruction is not None:
            sentences = [[instruction, sentence] for sentence in sentences]
        return sentences

    def encode(
        self,
        sentences: list[str],
        *,
        task: Optional[Task] = None,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> np.ndarray:
        if self.max_batch_size and batch_size > self.max_batch_size:
            batch_size = self.max_batch_size

        batched_embeddings = []
        if task is not None:
            instruction = task_to_instruction(task)
        else:
            instruction = None

        for batch in tqdm(batched(sentences, batch_size)):
            preprocessed_batch = self.preprocess(batch, instruction=instruction)
            with torch.inference_mode():
                embedded_batch = self.model.encode(preprocessed_batch)
            batched_embeddings.append(embedded_batch)

        return torch.cat(batched_embeddings).numpy()


@models.register("TTC-L2V-supervised-da-1")
def create_llm2vec_da_mntp_ttc_supervised() -> SebModel:
    base_model = "jealk/llm2vec-da-mntp"
    peft_model = "jealk/TTC-L2V-supervised-1"
    meta = ModelMeta(
        name="TTC-L2V-supervised-da-1",
        huggingface_name=peft_model,
        reference=f"https://huggingface.co/{peft_model}",
        languages=["da"],
        open_source=True,
        embedding_size=4096,
        architecture="LLM2Vec",
        release_date=date(2024, 12, 20),
    )
    partial_model = partial(
        LLM2VecModel,
        base_model_name_or_path=base_model,
        peft_model_name_or_path=peft_model,
        max_length=8192,
    )
    return SebModel(
        encoder=LazyLoadEncoder(partial_model),
        meta=meta,
    )


@models.register("TTC-L2V-unsupervised-da-1")
def create_llm2vec_da_mntp_ttc_unsupervised() -> SebModel:
    base_model = "jealk/llm2vec-da-mntp"
    peft_model = "jealk/TTC-L2V-unsupervised-1"
    meta = ModelMeta(
        name="TTC-L2V-unsupervised-da-1",
        huggingface_name=peft_model,
        reference=f"https://huggingface.co/{peft_model}",
        languages=["da"],
        open_source=True,
        embedding_size=4096,
        architecture="LLM2Vec",
        release_date=date(2024, 12, 20),
    )
    partial_model = partial(
        LLM2VecModel,
        base_model_name_or_path=base_model,
        peft_model_name_or_path=peft_model,
        max_length=8192,
    )
    return SebModel(
        encoder=LazyLoadEncoder(partial_model),
        meta=meta,
    )
