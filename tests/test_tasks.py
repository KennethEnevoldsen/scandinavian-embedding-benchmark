import pytest

import seb

all_tasks = seb.tasks.get_all()


@pytest.mark.parametrize("task_name", all_tasks.keys())
@pytest.mark.parametrize("model_name", "sentence-transformers/all-MiniLM-L6-v2")
def test_task(task_name: str, model_name: str):
    task: seb.Task = seb.get_task(task_name)
    model: seb.ModelInterface = seb.get_model(model_name)
    task_result = task.evaluate(model)
    assert isinstance(task_result, seb.TaskResult)
