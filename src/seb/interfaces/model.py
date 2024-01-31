import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol, runtime_checkable

from numpy.typing import ArrayLike
from pydantic import BaseModel

from seb.interfaces.language import Language

if TYPE_CHECKING:
    from .task import Task


@runtime_checkable
class Encoder(Protocol):
    """
    Interface which all models must implement.
    """

    def encode(
        self,
        sentences: list[str],
        *,
        task: Optional["Task"] = None,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> ArrayLike:
        """Returns a list of embeddings for the given sentences.

        Args:
            sentences: List of sentences to encode
            task: The task to encode for. This allows the model to encode differently for different tasks. Will always be given but does not need
                to be used.
            batch_size: Batch size for the encoding
            kwargs: arguments to pass to the models encode method

        Returns:
            Embeddings for the given documents
        """
        ...


class ModelMeta(BaseModel):
    """
    The metadata object for a model. This includes information such as the name, description, languages, etc.
    """

    name: str
    description: Optional[str] = None
    huggingface_name: Optional[str] = None
    reference: Optional[str] = None
    languages: list[Language] = []
    open_source: bool = False
    embedding_size: Optional[int] = None

    def get_path_name(self) -> str:
        if self.huggingface_name is None:
            return self._name_to_path(self.name)
        return self._name_to_path(self.huggingface_name)

    @staticmethod
    def _name_to_path(name: str) -> str:
        return name.replace("/", "__").replace(" ", "_")

    def get_huggingface_url(self) -> str:
        if self.huggingface_name is None:
            raise ValueError("This model does not have an associated huggingface name.")
        return f"https://huggingface.co/{self.huggingface_name}"

    def to_disk(self, path: Path) -> None:
        with path.open("w") as f:
            f.write(self.model_dump_json())

    @classmethod
    def from_disk(cls, path: Path) -> "ModelMeta":
        with path.open() as f:
            model_meta = json.load(f)
        return cls(**model_meta)


@dataclass
class LazyLoadEncoder(Encoder):
    """Encoder object, which lazy loads the model on the first call to encode()"""

    loader: Callable[[], Encoder]
    _model: Optional[Encoder] = None

    @property
    def model(self) -> Encoder:
        """
        Dynimically load the model.
        """
        if self._model is None:
            self._model = self.loader()
        return self._model

    def encode(
        self,
        sentences: list[str],
        *,
        task: Optional["Task"] = None,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> ArrayLike:
        """
        Returns a list of embeddings for the given sentences.
        Args:
            sentences: List of sentences to encode
            task: The task to encode for. This allows the model to encode differently for different tasks. Will always be given but does not need
                to be used.
            batch_size: Batch size for the encoding
            kwargs: arguments to pass to the models encode method

        Returns:
            Embeddings for the given documents
        """
        return self.model.encode(sentences, batch_size=batch_size, task=task, **kwargs)

    def encode_queries(self, queries: list[str], batch_size: int, **kwargs):  # noqa
        try:
            return self.model.encode_queries(queries, batch_size=batch_size, **kwargs)  # type: ignore
        except AttributeError:
            return self.encode(queries, task=None, batch_size=batch_size, **kwargs)

    def encode_corpus(self, corpus: list[dict[str, str]], batch_size: int, **kwargs):  # noqa
        try:
            return self.model.encode_corpus(corpus, batch_size=batch_size, **kwargs)  # type: ignore
        except AttributeError:
            sep = " "
            if isinstance(corpus, dict):
                sentences = [
                    (corpus["title"][i] + sep + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][i].strip()  # type: ignore
                    for i in range(len(corpus["text"]))  # type: ignore
                ]
            else:
                sentences = [(doc["title"] + sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus]
            return self.encode(sentences, task=None, batch_size=batch_size, **kwargs)


@dataclass
class SebModel:
    """
    An embedding model as implemented in SEB. It notably dynamically loads models (such that models are not loaded when a cache is hit)
    and includes metadata pertaining to the specific model.
    """

    meta: ModelMeta
    encoder: Encoder

    @property
    def number_of_parameters(self) -> Optional[int]:
        """
        Returns the number of parameters in the model.
        """
        if hasattr(self.encoder, "num_parameters"):
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)  # type: ignore
        return None
