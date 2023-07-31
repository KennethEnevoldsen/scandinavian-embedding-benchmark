from datetime import datetime
from pathlib import Path

import pytest

import seb

all_tasks = seb.get_all_tasks()
all_tasks_names = [
    task.name for task in all_tasks if not task.name.startswith("test ")
]  # ignore tasks intended for testing


@pytest.fixture
def task_result() -> seb.TaskResult:
    task_result = seb.TaskResult(
        task_name="test",
        scores={"en": {"test_measure": 0.4}, "da": {"test_measure": 0.2}},
        main_score="test_measure",
        time_of_run=datetime.now(),
        task_version="0.0.1",
        task_description="Just a test",
    )
    return task_result


def test_all_tasks_are_unique():
    all_tasks_names = [task.name for task in all_tasks]
    assert len(all_tasks_names) == len(set(all_tasks_names))


def test_task_name_is_registried_under_same_name():
    for regristry_name, task_create_fn in seb.tasks.get_all().items():
        task = task_create_fn()
        assert task.name == regristry_name


def test_read_write_tasks(task_result):
    tmp_path = Path("tests/tmpfiles/task_result.json")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)

    task_result.to_disk(tmp_path)
    # check that the file exists
    assert tmp_path.exists()
    task_result_from_disk = seb.TaskResult.from_disk(tmp_path)

    assert task_result_from_disk == task_result


def test_task_result_main_score(task_result: seb.TaskResult):
    assert task_result.get_main_score(["da"]) == 0.2
    assert task_result.get_main_score(["en"]) == 0.4
    assert (
        task_result.get_main_score() - task_result.get_main_score(["da", "en"]) < 0.0001
    )
    assert task_result.get_main_score(["da", "en"]) - 0.3 < 0.0001


@pytest.mark.skip(
    reason="This test downloads all datasets. It takes a long time to test and often fails due to errors on HF's side."
)
@pytest.mark.parametrize("task_name", all_tasks_names)
@pytest.mark.parametrize("model_name", ["sentence-transformers/all-MiniLM-L6-v2"])
def test_all_tasks(task_name: str, model_name: str):
    task: seb.Task = seb.get_task(task_name)
    model: seb.SebModel = seb.get_model(model_name)

    assert isinstance(task, seb.Task)
    assert isinstance(model, seb.SebModel)
    assert isinstance(model.model, seb.ModelInterface)

    task_result = task.evaluate(model)
    assert isinstance(task_result, seb.TaskResult)
