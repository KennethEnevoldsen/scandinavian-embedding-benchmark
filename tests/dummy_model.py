import numpy as np
import seb
from seb.registries import models


@models.register("test_model")
def create_test_model() -> seb.EmbeddingModel:
    class TestEncoder(seb.Encoder):
        def encode(
            self,
            sentences: list[str],
            batch_size: int,  # noqa: ARG002
            **kwargs: dict,  # noqa: ARG002
        ) -> np.ndarray:
            # create random array of 100, pr text
            return np.array([np.random.rand(100) for _ in sentences])

    def load_test_model() -> TestEncoder:
        return TestEncoder()

    return seb.EmbeddingModel(
        meta=seb.ModelMeta(name="test_model", embedding_size=100),
        loader=load_test_model,
    )
