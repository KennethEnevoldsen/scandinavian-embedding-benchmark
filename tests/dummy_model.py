from typing import List

import numpy as np

import seb
from seb.registries import models


@models.register("test_model")
def create_test_model() -> seb.SebModel:
    class TestEncoder:
        def encode(
            self, sentences: List[str], batch_size: int, **kwargs
        ) -> List[np.ndarray]:
            # create random array of 100, pr text
            return [np.random.rand(100) for _ in sentences]

    def load_test_model():
        return TestEncoder()

    assert isinstance(TestEncoder, seb.ModelInterface)

    return seb.SebModel(meta=seb.ModelMeta(name="test_model"), loader=load_test_model)  # type: ignore
