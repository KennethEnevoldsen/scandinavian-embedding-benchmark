"""
This section contains big tests which are too heavy to run as part of the CI, but are nice to have
"""

import pytest

import seb

all_models = seb.get_all_models()


# @pytest.mark.skip(reason="This test loads in all models. It is too heave to have running as a CI")
@pytest.mark.parametrize("model", all_models)
@pytest.mark.parametrize("task", [seb.get_task("DKHate")])
def test_model(model: seb.SebModel, task: seb.Task):
    """
    Test if the models encodes as expected
    """
    task.evaluate(model)
