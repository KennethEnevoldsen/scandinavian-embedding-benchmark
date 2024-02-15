from datetime import datetime
from typing import Any

import numpy as np
import seb
from seb.registries import models, tasks


@tasks.register("test-encode-task")
def create_test_encode_task() -> seb.Task:
    class DummyTask(seb.Task):
        name = "test-encode-task"
        main_score = "a_metric"
        description = "NA"
        reference = "NA"
        version = "NA"
        languages = []  # noqa: RUF012
        domain = []  # noqa: RUF012
        task_type = "Classification"

        def evaluate(self, model: seb.Encoder) -> seb.TaskResult:
            model.encode(["a test sentence"], task=self)

            return seb.TaskResult(
                task_name="test-encode-task",
                task_description="NA",
                task_version="NA",
                time_of_run=datetime.now(),
                scores={"en": {"a_metric": 1.0}},
                main_score="a_metric",
            )

        def get_descriptive_stats(self) -> dict[str, Any]:
            return {}

    return DummyTask()


@models.register("test_model")
def create_test_model() -> seb.Encoder:
    class TestEncoder:
        def encode(
            self,
            sentences: list[str],
            batch_size: int = 32,  # noqa: ARG002
            **kwargs: dict,  # noqa: ARG002
        ) -> np.ndarray:
            # create random array of 100, pr text
            return np.array([np.random.rand(100) for _ in sentences])

    def load_test_model() -> TestEncoder:
        return TestEncoder()

    assert isinstance(TestEncoder, seb.Encoder)

    return seb.SebModel(
        meta=seb.ModelMeta(name="test_model", embedding_size=100),
        encoder=seb.LazyLoadEncoder(load_test_model),  # type: ignore
    )
