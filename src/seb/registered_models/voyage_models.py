"""
The openai embedding api's evaluated on the SEB benchmark.
"""

from __future__ import annotations

import logging
import time
from datetime import date
from functools import partial, wraps
from typing import Any, Literal

import numpy as np

from seb.interfaces.model import Encoder, LazyLoadEncoder, ModelMeta, SebModel
from seb.registries import models

logger = logging.getLogger(__name__)


def token_limit(max_tpm: int, interval: int = 60):  # noqa
    limit_interval_start_ts = time.time()
    used_tokens = 0

    def decorator(func):  # noqa
        @wraps(func)
        def wrapper(*args, **kwargs):  # noqa
            nonlocal limit_interval_start_ts, used_tokens

            result = func(*args, **kwargs)
            used_tokens += result.total_tokens

            current_time = time.time()
            if current_time - limit_interval_start_ts > interval:
                limit_interval_start_ts = current_time
                used_tokens = 0

            if used_tokens > max_tpm:
                time.sleep(interval - (current_time - limit_interval_start_ts))
                used_tokens = 0
            return result

        return wrapper

    return decorator


def rate_limit(max_rpm: int, interval: int = 60):  # noqa
    request_interval = interval / max_rpm
    previous_call_ts: float | None = None

    def decorator(func):  # noqa
        @wraps(func)
        def wrapper(*args, **kwargs):  # noqa
            current_time = time.time()
            nonlocal previous_call_ts
            if previous_call_ts is not None and current_time - previous_call_ts < request_interval:
                time.sleep(request_interval - (current_time - previous_call_ts))

            result = func(*args, **kwargs)
            previous_call_ts = time.time()
            return result

        return wrapper

    return decorator


class VoyageWrapper(Encoder):
    def __init__(
        self,
        model_name: str,
        max_retries: int = 5,
        max_rpm: int = 300,
        max_tpm: int = 120_000,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        try:
            import voyageai  # type: ignore
        except ImportError as e:
            raise ImportError("Please install voyageai to use this model using `pip install 'seb[voyageai]'`") from e

        self._client = voyageai.Client(max_retries=max_retries)
        self._embed_func = rate_limit(max_rpm)(token_limit(max_tpm)(self._client.embed))
        self._model_name = model_name
        self._max_tpm = max_tpm
        self.sep = " "

    def encode(self, sentences: list[str], *, batch_size: int = 32, **kwargs: Any) -> np.ndarray:  # noqa: ARG002
        return self._batched_encode(sentences, batch_size, "document")

    def encode_queries(self, queries: list[str], *, batch_size: int = 32, **kwargs: Any) -> np.ndarray:  # noqa: ARG002
        return self._batched_encode(queries, batch_size, "query")

    def encode_corpus(
        self,
        corpus: list[dict[str, str]] | dict[str, list[str]],
        *,
        batch_size: int = 32,
        **kwargs: Any,  # noqa: ARG002
    ) -> np.ndarray:
        if isinstance(corpus, dict):
            sentences = [
                (corpus["title"][i] + self.sep + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][i].strip()  # type: ignore
                for i in range(len(corpus["text"]))  # type: ignore
            ]
        else:
            sentences = [(doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus]

        return self._batched_encode(sentences, batch_size, "document")

    def _batched_encode(
        self,
        sentences: list[str],
        batch_size: int,
        input_type: Literal["query", "document"],
    ) -> np.ndarray:
        embeddings, index = [], 0

        while index <= len(sentences) - 1:
            batch, batch_tokens = [], 0
            while index < len(sentences) and len(batch) < batch_size and batch_tokens < self._max_tpm:
                n_tokens = len(self._client.tokenize([sentences[index]], model=self._model_name)[0])
                if batch_tokens + n_tokens > self._max_tpm:
                    break
                batch_tokens += n_tokens
                batch.append(sentences[index])
                index += 1

            embeddings.extend(
                self._embed_func(
                    texts=batch,
                    model=self._model_name,
                    input_type=input_type,
                    truncation=True,
                ).embeddings
            )

        return np.array(embeddings)


@models.register("voyage-multilingual-2")
def create_voyage_multilingual_2() -> SebModel:
    api_name = "voyage-multilingual-2"
    meta = ModelMeta(
        name=api_name,
        huggingface_name=None,
        reference="https://blog.voyageai.com/2024/06/10/voyage-multilingual-2-multilingual-embedding-model/",
        languages=[],
        open_source=False,
        embedding_size=1024,
        architecture="API",
        release_date=date(2024, 6, 10),
    )
    return SebModel(
        encoder=LazyLoadEncoder(partial(VoyageWrapper, model_name=api_name)),  # type: ignore
        meta=meta,
    )


if __name__ == "__main__":
    model = create_voyage_multilingual_2()
    test = model.encoder.encode(["Hello world"])
