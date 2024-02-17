from datetime import datetime

import pytest
import seb
from seb.cli import run_benchmark_cli
from seb.registries import tasks


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

        def get_descriptive_stats(self) -> dict:
            return {}

        def get_documents(self) -> list[str]:
            return []

    return DummyTask()


@pytest.mark.parametrize(
    "model",
    [
        "intfloat/e5-small",
    ],
)
def test_integration_dummy(model: str):
    """Runs all sorts of models on a dummy task to see if they can run without breaking.
    Cache is ignored so that the models are actually run.
    """
    tasks = ["test-encode-task"]
    run_benchmark_cli(models=[model], tasks=tasks, ignore_cache=True)


@pytest.mark.parametrize(
    "model",
    [
        "sentence-transformers/all-MiniLM-L6-v2",
    ],
)
def test_integration_lcc(model: str):
    """Runs model(s) on LCC to see if everything works in order with tasks included."""
    tasks = ["LCC"]
    run_benchmark_cli(models=[model], tasks=tasks, ignore_cache=True)
