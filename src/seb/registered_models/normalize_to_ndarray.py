from typing import Union

import numpy as np
import torch


def normalize_to_ndarray(embeddings: Union[torch.Tensor, np.ndarray, list[np.ndarray], list[torch.Tensor]]) -> np.ndarray:
    if isinstance(embeddings, list):
        if isinstance(embeddings[0], torch.Tensor):
            return torch.cat(embeddings).detach().cpu().numpy()  # type: ignore
        return np.concatenate(embeddings)
    if isinstance(embeddings, torch.Tensor):
        return embeddings.detach().cpu().numpy()
    return embeddings
