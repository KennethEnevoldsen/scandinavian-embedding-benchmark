from datetime import datetime

import seb
from seb.registries import tasks


@tasks.register("test task")
def create_test_task() -> seb.Task:
    class DummyTask(seb.Task):
        name = "test task"
        main_score = "a_metric"
        description = "NA"
        reference = "NA"
        version = "NA"
        languages = []

        def evaluate(self, model: seb.ModelInterface) -> seb.TaskResult:
            return seb.TaskResult(
                task_name="test task",
                task_description="NA",
                task_version="NA",
                time_of_run=datetime.now(),
                scores={"en": {"a_metric": 1.0}},
                main_score="a_metric",
            )

    return DummyTask()


@tasks.register("test encode task")
def create_test_encode_task() -> seb.Task:
    class DummyTask(seb.Task):
        name = "test encode task"
        main_score = "a_metric"
        description = "NA"
        reference = "NA"
        version = "NA"
        languages = []

        def evaluate(self, model: seb.ModelInterface) -> seb.TaskResult:
            model.encode(["a test sentence"])

            return seb.TaskResult(
                task_name="test task",
                task_description="NA",
                task_version="NA",
                time_of_run=datetime.now(),
                scores={"en": {"a_metric": 1.0}},
                main_score="a_metric",
            )

    return DummyTask()


@tasks.register("test raise error task")
def create_test_raise_error_task() -> seb.Task:
    class DummyTask(seb.Task):
        name = "test raise error task"
        main_score = "a_metric"
        description = "NA"
        reference = "NA"
        version = "NA"
        languages = []

        def evaluate(self, model: seb.ModelInterface) -> seb.TaskResult:
            raise ValueError("Test raised error. This error should be handled.")

    return DummyTask()
