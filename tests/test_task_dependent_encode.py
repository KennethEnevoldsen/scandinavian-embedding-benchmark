from datetime import datetime
from pathlib import Path

import numpy as np
import seb

from .dummy_task import TestTask


def create_test_model_with_task_dependent_encode() -> seb.SebModel:
    class TestEncoder(seb.Encoder):
        def encode(
            self,
            sentences: list[str],
            *,
            task: seb.Task,
            batch_size: int = 32,  # noqa: ARG002
        ) -> np.ndarray:
            if task.task_type == "SNS":
                return np.array([np.ones(100) for _ in sentences])
            return np.array([np.zeros(100) for _ in sentences])

    def load_test_model() -> TestEncoder:
        return TestEncoder()

    return seb.SebModel(
        meta=seb.ModelMeta(name="test_model_with_task_dependent_encode", embedding_size=100),
        encoder=seb.LazyLoadEncoder(load_test_model),
    )


def create_all_is_0_task() -> seb.Task:
    class TestTaskAllEmbeddingIsOne(TestTask):
        name = "embeddings is one task"
        task_type: str = "SNS"

        def evaluate(self, model: seb.Encoder) -> seb.TaskResult:
            out = model.encode(["a test sentence"], task=self)
            assert np.all(out == 1)

            return seb.TaskResult(
                task_name=self.name,
                task_description="NA",
                task_version="NA",
                time_of_run=datetime.now(),
                scores={"nb": {"a_metric": 1.0}},
                main_score="a_metric",
            )

    return TestTaskAllEmbeddingIsOne()


def create_all_is_1_task() -> seb.Task:
    class TestTaskAllEmbeddingIsZero(TestTask):
        name = "all embeddings is 0 task"
        task_type = "Classification"

        def evaluate(self, model: seb.Encoder) -> seb.TaskResult:
            out = model.encode(["a test sentence"], task=self)
            assert np.all(out == 0)

            return seb.TaskResult(
                task_name=self.name,
                task_description="NA",
                task_version="NA",
                time_of_run=datetime.now(),
                scores={"nb": {"a_metric": 1.0}},
                main_score="a_metric",
            )

    return TestTaskAllEmbeddingIsZero()


def test_task_dependent_encode(tmp_path: Path):
    model = create_test_model_with_task_dependent_encode()

    tasks = [
        create_all_is_0_task(),
        create_all_is_1_task(),
    ]

    benchmark = seb.Benchmark(tasks=tasks)
    result = benchmark.evaluate_model(model, cache_dir=tmp_path)
    assert result.get_main_score() == 1, "both datasets should have score of 1 if they run successfully"
