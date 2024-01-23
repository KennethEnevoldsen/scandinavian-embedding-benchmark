from datetime import datetime
from typing import Any

import seb


class TestTask(seb.Task):
    name = "test task"
    main_score = "a_metric"
    description = "NA"
    reference = "NA"
    version = "NA"
    languages = []  # noqa: RUF012
    domain = []  # noqa: RUF012
    task_type = "Classification"

    def evaluate(self, model: seb.Encoder) -> seb.TaskResult:  # noqa: ARG002
        return seb.TaskResult(
            task_name="test task",
            task_description="NA",
            task_version="NA",
            time_of_run=datetime.now(),
            scores={"nb": {"a_metric": 1.0}},
            main_score="a_metric",
        )

    def get_descriptive_stats(self) -> dict[str, Any]:
        return {}


def create_test_task() -> seb.Task:
    return TestTask()  # type: ignore


def create_test_encode_task() -> seb.Task:
    class TestTaskWithEncode(TestTask):
        name = "test encode task"

        def evaluate(self, model: seb.Encoder) -> seb.TaskResult:
            model.encode(["a test sentence"], task=self)

            return seb.TaskResult(
                task_name="test task",
                task_description="NA",
                task_version="NA",
                time_of_run=datetime.now(),
                scores={"nb": {"a_metric": 1.0}},
                main_score="a_metric",
            )

    return TestTaskWithEncode()


def create_test_raise_error_task() -> seb.Task:
    """
    Note this task is not registered as it will cause errrors in other tests.
    """

    class TestTaskWithError(TestTask):
        name = "test raise error task"

        def evaluate(self, model: seb.Encoder) -> seb.TaskResult:  # noqa ARG002
            raise ValueError("Test raised error. This error should be handled.")

    return TestTaskWithError()
