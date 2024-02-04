"""
This section contains big tests which are too heavy to run as part of the CI, but are nice to have
"""

import pytest
import seb

from .dummy_task import create_test_encode_task

all_models = seb.get_all_models()
openai_models = []


@pytest.mark.skip(
    reason="This test loads in all models. It is too heavy to have running as a CI",
)
@pytest.mark.parametrize("model", all_models)
@pytest.mark.parametrize("task", [create_test_encode_task()])
def test_model(model: seb.SebModel, task: seb.Task):
    """
    Test if the models encodes as expected
    """
    task.evaluate(model.encoder)


@pytest.mark.skip(
    reason="This test loads in all models. It is too heavy to have running as a CI",
)
@pytest.mark.parametrize("model", all_models)
def test_embedding_match_what_is_stated(model: seb.SebModel):
    """
    This test checks if the embedding size matches what is stated in the ModelMeta
    """
    output = model.encoder.encode(["test"])
    output_embedding_size = output.shape[1]
    assert output_embedding_size == model.meta.embedding_size


@pytest.mark.skip(
    reason="This test applied the openai embedding models. It is too expensive to have running as a CI",
)
@pytest.mark.parametrize("model", [seb.get_model("text-embedding-ada-002")])
@pytest.mark.parametrize("task", [create_test_encode_task()])
def test_openai_model(model: seb.SebModel, task: seb.Task):
    """
    Test if the models encodes as expected
    """
    task.evaluate(model.encoder)
