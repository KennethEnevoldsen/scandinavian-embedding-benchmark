from typing import List, Optional

import pytest
from typer.testing import CliRunner

from seb.cli import app

runner = CliRunner()


@pytest.mark.parametrize(
    "model_names, languages, tasks, categories",
    [
        (["Maltehb/aelaectra-danish-electra-small-cased"], ["da"], None, None),
        (
            ["sentence-transformers/all-mpnet-base-v2"],
            None,
            ["LccSentimentClassification", "DKHateClassification"],
            None,
        ),
        (["sentence-transformers/all-mpnet-base-v2"], None, None, ["sentiment"]),
        (["sentence-transformers/all-mpnet-base-v2"], None, None, ["bitext"]),
        (["sentence-transformers/all-mpnet-base-v2", "Maltehb/aelaectra-danish-electra-small-cased"], None, None, ["bitext"]),
    ],
)
def test_app(model_names: List[str], languages: Optional[List[str]], tasks: Optional[List[str]], categories: Optional[List[str]] = None):
    cmd = ["seb"]
    for mdl in model_names:
        cmd.append("--model_name")
        cmd.append(mdl)
    for lang in languages:
        cmd.append("--lang")
        cmd.append(lang)
    for task in tasks:
        cmd.append("--task")
        cmd.append(task)
    for cat in categories:
        cmd.append("--category")
        cmd.append(cat)


    result = runner.invoke(app, cmd)
    assert result.exit_code == 0