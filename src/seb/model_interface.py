from typing import Callable, List, Optional, Protocol, Union, runtime_checkable

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
        self, sentences: List[str], batch_size: int = 32, **kwargs
    ) -> List[ArrayLike]:
        """Returns a list of embeddings for the given sentences.
        Args:
            sentences: List of sentences to encode
            batch_size: Batch size for the encoding

        Returns:
            List of embeddings for the given sentences
        """
        ...


class ModelMeta(BaseModel):
    name: str
    description: Optional[str] = None
    huggingface_name: Optional[str] = None
    reference: Optional[str] = None
    languages: List[str] = []

    def get_path_name(self):
        if self.huggingface_name is None:
            return name_to_path(self.name)
        return name_to_path(self.huggingface_name)

    def get_huggingface_url(self) -> str:
        if self.huggingface_name is None:
            raise ValueError("This model does not have an associated huggingface name.")
        return f"https://huggingface.co/{self.huggingface_name}"


class SebModel(BaseModel):
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
        else:
            return None

    def encode(
        self, sentences: List[str], batch_size: int = 32, **kwargs
    ) -> List[ArrayLike]:
        """
        Returns a list of embeddings for the given sentences.
        Args:
            sentences: List of sentences to encode
            batch_size: Batch size for the encoding

        Returns:
            List of embeddings for the given sentences
        """
        return self.model.encode(sentences, batch_size=batch_size, **kwargs)
