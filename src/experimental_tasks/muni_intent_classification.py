"""
To run this task:

seb run -t MuniIntent -m multilingual-e5-large-instruct,multilingual-e5-large,paraphrase-multilingual-mpnet-base-v2 -c src/experimental_tasks/muni_intent_classification.py
seb run -t MuniIntent -m paraphrase-multilingual-mpnet-base-v2 -c src/experimental_tasks/muni_intent_classification.py
"""

from typing import Any

from mteb.abstasks import AbsTaskClassification
from seb.interfaces.mteb_task import MTEBTask
from seb.interfaces.task import Task
from seb.registries import tasks


class MuniIntentClassification(AbsTaskClassification):
    @property
    def description(self) -> dict[str, Any]:
        return {
            "name": "MuniIntentClassification",
            "hf_hub_name": "KennethEnevoldsen/muni_intents",
            "description": "An intent classification dataset used for the MUNI chatbot.",
            "reference": "NA",
            "type": "Classification",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["da"],  # assumed to be bokmål
            "main_score": "accuracy",
            "n_experiments": 10,
            "samples_per_label": 16,
            "revision": "f342548ed8ed4e1b3a31c546a34d2f6fd1a6ed12",
        }


@tasks.register("MuniIntent")
def create_swedn_sts() -> Task:
    task = MTEBTask(MuniIntentClassification())
    task.name = "MuniIntent"
    task.version = "0.0.1"
    task.domain = ["non-fiction"]
    return task
