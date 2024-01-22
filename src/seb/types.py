from typing import Literal, TypedDict, Union

from numpy import ndarray
from torch import Tensor

ArrayLike = Union[ndarray, Tensor]


Domain = Literal[
    "social",
    "poetry",
    "wiki",
    "fiction",
    "non-fiction",
    "web",
    "legal",
    "news",
    "academic",
    "spoken",
    "reviews",
    "blog",
    "medical",
    "government",
    "bible",
]

Language = Literal["da", "nb", "nn", "sv", "da-bornholm", "is", "fo"]
languages_in_seb: list[Language] = ["da", "nb", "nn", "sv", "da-bornholm", "is", "fo"]

TaskType = Literal["Classification", "Retrieval", "STS", "BitextMining", "Clustering", "Speed"]


class DescriptiveDatasetStats(TypedDict):
    mean_document_length: float
    std_document_length: float
    num_documents: int
