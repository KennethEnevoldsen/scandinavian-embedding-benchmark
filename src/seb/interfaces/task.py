from typing import Any, Callable, Optional, Protocol, runtime_checkable

from pydantic import BaseModel

from ..result_dataclasses import TaskResult
from ..types import ArrayLike, DescriptiveDatasetStats, Domain, Language, TaskType


@runtime_checkable
class Task(Protocol):
    """
    A task is a specific evaluation task for a sentence embedding model.

    Attributes:
        name: The name of the task.
        main_score: The main score of the task.
        description: A description of the task.
        reference: A reference to the task.
        version: The version of the task.
        languages: The languages of the task.
        domain: The domains of the task. Should be one of the categories listed on https://universaldependencies.org
    """

    name: str
    main_score: str
    description: str
    reference: str
    version: str
    languages: list[Language]
    domain: list[Domain]
    task_type: TaskType

    def evaluate(self, model: "Encoder") -> TaskResult:
        """
        Evaluates a Sentence Embedding Model on the task.

        Args:
            model: A sentence embedding model.

        Returns:
            A TaskResult object.
        """
        ...

    def get_descriptive_stats(self) -> DescriptiveDatasetStats:
        ...

    def name_to_path(self) -> str:
        """
        Convert a name to a path.
        """
        name = self.name.replace("/", "__").replace(" ", "_")
        return name


@runtime_checkable
class Encoder(Protocol):
    """
    Interface which all models must implement.
    """

    def encode(
        self,
        sentences: list[str],
        task: Task,
        batch_size: int = 32,
        **kwargs: dict,
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
    name: str
    description: Optional[str] = None
    huggingface_name: Optional[str] = None
    reference: Optional[str] = None
    languages: list[str] = []
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


class EmbeddingModel(BaseModel):
    """
    An embedding model as implemented in SEB. It notably dynamically loads models (such that models are not loaded when a cache is hit)
    and includes metadata pertaining to the specific model.
    """

    meta: ModelMeta
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

    @property
    def number_of_parameters(self) -> Optional[int]:
        """
        Returns the number of parameters in the model.
        """
        if hasattr(self.model, "num_parameters"):
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)  # type: ignore
        return None

    def encode(
        self,
        sentences: list[str],
        batch_size: int = 32,
        **kwargs: Any,
    ) -> ArrayLike:
        """
        Returns a list of embeddings for the given sentences.
        Args:
            sentences: List of sentences to encode
            batch_size: Batch size for the encoding
            kwargs: arguments to pass to the models encode method

        Returns:
            Embeddings for the given documents
        """
        return self.model.encode(sentences, batch_size=batch_size, **kwargs)
