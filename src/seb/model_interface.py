from typing import Any, Callable, Optional, Protocol, Union, runtime_checkable

from numpy import ndarray
from pydantic import BaseModel
from torch import Tensor

from .utils import name_to_path

ArrayLike = Union[ndarray, Tensor]


@runtime_checkable
class ModelInterface(Protocol):
    """
    Interface which all models must implement.
    """

    def encode(
        self,
        sentences: list[str],
        batch_size: int = 32,
        **kwargs: dict,
    ) -> ArrayLike:
        """Returns a list of embeddings for the given sentences.
        Args:
            sentences: List of sentences to encode
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
            return name_to_path(self.name)
        return name_to_path(self.huggingface_name)

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
    loader: Callable[[], ModelInterface]
    _model: Optional[ModelInterface] = None

    @property
    def model(self) -> ModelInterface:
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
