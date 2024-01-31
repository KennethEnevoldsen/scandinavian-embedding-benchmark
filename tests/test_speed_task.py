import pytest
import seb
from seb.registered_tasks.speed import CPUSpeedTask, GPUSpeedTask

from .dummy_model import create_test_model


@pytest.mark.parametrize("model", [create_test_model(), "sentence-transformers/all-MiniLM-L6-v2"])
def test_cpu_speed_task(model: seb.SebModel):
    task = CPUSpeedTask()  # type: ignore
    model = create_test_model()

    result = task.evaluate(model.encoder)

    speed = result.get_main_score()
    assert speed > 0
    assert speed < 1000


@pytest.mark.skip(reason="GPU speed task is not available during CI")
@pytest.mark.parametrize("model", ["sentence-transformers/all-MiniLM-L6-v2"])
def test_gpu_speed_task(model: seb.SebModel):
    task = GPUSpeedTask()  # type: ignore
    model = create_test_model()

    result = task.evaluate(model.encoder)

    speed = result.get_main_score()
    assert speed > 0
    assert speed < 1000
