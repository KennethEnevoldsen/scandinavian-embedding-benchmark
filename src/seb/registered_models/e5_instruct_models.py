import logging
from collections.abc import Iterable, Sequence
from datetime import date
from functools import partial
from itertools import islice
from typing import Any, Literal, Optional, TypeVar

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BatchEncoding
from transformers.modeling_outputs import ModelOutput

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



def task_to_instruction_da(task: Task) -> str:
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
            "LCC": "Klassificér tekster baseret på deres sentiment",
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


def task_to_instruction_da(task: Task) -> str:
    if task.task_type in ["STS"]:
        return "Find semantisk lignende tekst"
    if task.task_type in ["Summarization"]:
        return "Givet et nyhedssammendrag, find andre semantisk lignende sammendrag"
    if task.task_type in ["BitextMining"]:
        task_name_to_instruct: dict[str, str] = {
            "Bornholm Parallel": "retrieve parallelle sætninger på dansk og bornholmsk",
            "Norwegian courts": "Find parallelle sætninger på norsk bokmål og nynorsk",
        }
        default_instruction = "Find parallelle sætninger."
        return task_name_to_instruct.get(task.name, default_instruction)
    if task.task_type in ["Classification"]:
        task_name_to_instruct: dict[str, str] = {
            "Angry Tweets": "Klassificér danske tweets efter sentiment. (positivt, negativt, neutralt)",
            "DKHate": "Klassificér danske tweets baseret på stødende indhold (stødende, ikke stødende)",
            "Da Political Comments": "Klassificér danske politiske kommentarer for sentiment",
            "DaLAJ": "Klassificér tekster baseret på sproglig acceptabilitet på svensk",
            "LCC": "Klassificér tekster baseret på sentiment",
            "Language Identification": "Klassificér tekster baseret på sprog",
            "Massive Intent": "Givet en brugerudtalelse som forespørgsel, find brugerens intentioner",
            "Massive Scenario": "Givet en brugerudtalelse som forespørgsel, find brugerens scenarier",
            "NoReC": "Klassificér norske anmeldelser efter sentiment",
            "SweReC": "Klassificér svenske anmeldelser efter sentiment",
            "Norwegian parliament": "Klassificér taler i det norske parlament baseret på politisk tilhørsforhold",
            "ScaLA": "Klassificér passager på skandinaviske sprog baseret på sproglig acceptabilitet",
        }
        default_instruction = "Klassificér brugerpassager"
        return task_name_to_instruct.get(task.name, default_instruction)
    if task.task_type in ["Clustering"]:
        task_name_to_instruct: dict[str, str] = {
            "ArxivClusteringP2P": "Identificer hoved- og sekundærkategorien af Arxiv-papirer baseret på titler og resuméer",
            "VG Clustering": "Identificer kategorierne (f.eks. sport) af givne artikler på norsk",
            "SNL Clustering": "Identificer kategorier i et norsk leksikon",
            "SwednClustering": "Identificer nyhedskategorier i svenske passager",
        }
        default_instruction = "Identificer kategorier i brugerpassager"
        return task_name_to_instruct.get(task.name, default_instruction)
    if task.task_type in ["Reranking"]:
        return "Find semantisk lignende passager."
    if task.task_type in ["Retrieval"]:
        task_name_to_instruct: dict[str, str] = {
            "Twitterhjerne": "Find svar på spørgsmål stillet i danske tweets",
            "SwednRetrieval": "Givet en svensk nyhedsoverskrift find resuméer eller nyhedsartikler",
            "TV2Nord Retrieval": "Givet et resumé af en dansk nyhedsartikel find den tilsvarende nyhedsartikel",
            "DanFEVER": "Givet en påstand på dansk, find dokumenter der understøtter påstanden",
            "SNL Retrieval": "Givet en leksikonoverskrift på norsk, find dens artikel",
            "NorQuad": "Givet et spørgsmål på norsk, find svaret i Wikipedia-artikler",
            "SweFAQ": "Find svar givet spørgsmål på svensk",
            "ArguAna": "Givet en påstand, find dokumenter der modbeviser påstanden",
            "ClimateFEVER": "Givet en påstand om klimaændringer, find dokumenter der understøtter eller modbeviser påstanden",
        }
        default_instruction = "Find tekst baseret på brugerforespørgsel."
        return task_name_to_instruct.get(task.name, default_instruction)
    return ""



class E5Instruct(Encoder):
    def __init__(self, model_name: str, max_length: int, max_batch_size: Optional[int] = None, **kwargs: Any):
        logger.info("Started loading e5 instruct model")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, **kwargs)
        self.max_length = max_length
        self.max_batch_size = max_batch_size
        self.task_to_instruction = task_to_instruction

    def preprocess(self, sentences: Sequence[str], instruction: str, encode_type: EncodeTypes) -> BatchEncoding:
        if encode_type == "query":
            sentences = [f"Instruction: {instruction}\nQuery: {sentence}" for sentence in sentences]

        batch_dict = self.tokenizer(
            sentences,  # type: ignore
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        return batch_dict.to(self.model.device)

    def get_embedding_from_output(self, output: ModelOutput, batch_dict: BatchEncoding) -> torch.Tensor:
        return self.average_pool(output.last_hidden_state, batch_dict["attention_mask"])  # type: ignore

    @staticmethod
    def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def encode(
        self,
        sentences: list[str],
        *,
        task: Optional[Task] = None,
        batch_size: int = 32,
        encode_type: EncodeTypes = "query",
        **kwargs: Any,  # noqa
    ) -> np.ndarray:
        if self.max_batch_size and batch_size > self.max_batch_size:
            batch_size = self.max_batch_size
        batched_embeddings = []
        if task is not None:  # noqa
            instruction = self.task_to_instruction(task)
        else:
            instruction = ""
        for batch in tqdm(batched(sentences, batch_size)):
            with torch.inference_mode():
                batch_dict = self.preprocess(batch, instruction=instruction, encode_type=encode_type)
                outputs = self.model(**batch_dict)
                embeddings = self.get_embedding_from_output(outputs, batch_dict)
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
    
    def to(self, device: torch.device):
        self.model.to(device)



class E5Mistral(E5Instruct):
    def __init__(self):
        super().__init__("intfloat/e5-mistral-7b-instruct", max_length=4096, max_batch_size=4, torch_dtype=torch.float16)

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

    def get_embbeding_from_output(self, output: ModelOutput, batch_dict: BatchEncoding) -> torch.Tensor:
        return self.last_token_pool(output.last_hidden_state, batch_dict["attention_mask"])  # type: ignore

    def preprocess(self, sentences: Sequence[str], instruction: str, encode_type: EncodeTypes) -> BatchEncoding:
        if encode_type == "query":
            sentences = [f"Instruction: {instruction}\nQuery: {sentence}" for sentence in sentences]
        batch_dict: BatchEncoding = self.tokenizer(
            sentences,  # type: ignore
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


class E5PEFTInstruct(E5Mistral):
    def __init__(self, model_name: str, max_length: int = 4096, max_batch_size: Optional[int] = 4, **kwargs: Any):
        from peft import PeftConfig, PeftModel  # type: ignore

        repo_id = model_name
        config = PeftConfig.from_pretrained(repo_id)

        base_model = AutoModel.from_pretrained(config.base_model_name_or_path, torch_dtype=torch.float16, device_map="auto")
        self.model = PeftModel.from_pretrained(base_model, repo_id)
        self.tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-mistral-7b-instruct")
        self.model = AutoModel.from_pretrained(model_name, **kwargs)
        self.max_length = max_length
        self.max_batch_size = max_batch_size

        self.task_to_instruction = task_to_instruction_da


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
        architecture="Mistral",
        release_date=date(2023, 12, 20),
    )
    return SebModel(
        encoder=LazyLoadEncoder(E5Mistral),
        meta=meta,
    )


@models.register("multilingual-e5-large-instruct")
def create_multilingual_e5_large_instruct() -> SebModel:
    hf_name = "intfloat/multilingual-e5-large-instruct"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=[],
        open_source=True,
        embedding_size=1024,
        architecture="XLM-R",
        release_date=date(2023, 12, 20),
    )
    partial_model = partial(E5Instruct, model_name=hf_name, max_length=512)
    return SebModel(
        encoder=LazyLoadEncoder(partial_model),
        meta=meta,
    )


@models.register("e5-munin-neuralbeagle")
def create_e5_munin_neuralbeagle() -> SebModel:
    hf_name = "KennethEnevoldsen/munin-neuralbeagle-7b-e5"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=[],
        open_source=True,
        embedding_size=1024,
        architecture="Mistral",
        release_date=date(2023, 2, 27),
    )
    partial_model = partial(E5PEFTInstruct, model_name=hf_name)
    return SebModel(
        encoder=LazyLoadEncoder(partial_model),
        meta=meta,
    )


@models.register("e5-munin")
def create_e5_munin() -> SebModel:
    hf_name = "KennethEnevoldsen/munin-7b-e5"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=[],
        open_source=True,
        embedding_size=1024,
        architecture="Mistral",
        release_date=date(2023, 2, 27),
    )
    partial_model = partial(E5PEFTInstruct, model_name=hf_name)
    return SebModel(
        encoder=LazyLoadEncoder(partial_model),
        meta=meta,
    )
