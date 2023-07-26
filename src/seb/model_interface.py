from typing import Callable, List, Optional, Protocol, Union

from numpy import ndarray
from pydantic import BaseModel
from torch import Tensor

from .utils import name_to_path

ArrayLike = Union[ndarray, Tensor]


class ModelInterface(Protocol):
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
    model_name: str
    model_description: Optional[str] = None
    huggingface_name: Optional[str] = None
    model_reference: Optional[str] = None
    languages: List[str] = []

    def get_path_name(self):
        if self.huggingface_name is None:
            return name_to_path(self.model_name)
        return name_to_path(self.huggingface_name)


class SebModel(BaseModel):
    model_meta: ModelMeta
    model_loader: Callable[[], ModelInterface]
    _model: Optional[ModelInterface] = None

    @property
    def model(self) -> ModelInterface:
        """
        Dynimically load the model.
        """
        if self._model is None:
            self._model = self.model_loader()
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
