from typing import Any

from seb.interfaces.mteb_task import MTEBTask
from seb.interfaces.task import Task
from seb.mteb_tasks import NorwegianCourtsBitextMining, NorwegianParliamentClassification
from seb.registries import tasks


@tasks.register("NoReC")
def create_norec() -> Task:
    from mteb import NoRecClassification

    class NoRecClassificationCustom(NoRecClassification):
        @property
        def description(self) -> dict[str, Any]:
            return {
                "name": "NoRecClassification",
                "hf_hub_name": "mteb/norec_classification",
                "description": "A Norwegian dataset for sentiment classification on review",
                "reference": "https://aclanthology.org/L18-1661/",
                "type": "Classification",
                "category": "s2s",
                "eval_splits": ["test"],
                "eval_langs": ["nb"],
                "main_score": "accuracy",
                "n_experiments": 10,
                "samples_per_label": 16,
            }

    task = MTEBTask(NoRecClassificationCustom())
    task.name = "NoReC"
    task.domain = ["reviews"]
    task.task_subtypes = ["Sentiment Classification"]
    return task


@tasks.register("Norwegian parliament")
def create_norwegian_parliament() -> Task:
    task = MTEBTask(NorwegianParliamentClassification())
    task.name = "Norwegian parliament"
    task.domain = ["spoken"]
    task.task_subtypes = ["Political Classification"]
    return task


@tasks.register("Norwegian courts")
def create_norwegian_courts() -> Task:
    task = MTEBTask(NorwegianCourtsBitextMining())
    task.name = "Norwegian courts"
    task.domain = ["legal", "non-fiction"]
    task._text_columns = ["sentence1", "sentence2"]
    task.task_subtypes = ["Written form Pairing"]
    return task


@tasks.register("VG Clustering")
def create_vg_clustering() -> Task:
    from seb.mteb_tasks import VGClustering

    task = MTEBTask(VGClustering())
    task.name = "VG Clustering"
    task.version = "0.0.1"
    task.domain = ["non-fiction", "news"]
    task._text_columns = ["sentences"]
    task.task_subtypes = ["Thematic Clustering"]
    return task


@tasks.register("SNL Clustering")
def create_sts_clustering() -> Task:
    from seb.mteb_tasks import SNLClustering

    task = MTEBTask(SNLClustering())
    task.name = "SNL Clustering"
    task.version = "0.0.1"
    task.domain = ["non-fiction", "wiki"]
    task._text_columns = ["sentences"]
    task.task_subtypes = ["Thematic Clustering"]
    return task


@tasks.register("SNL Retrieval")
def create_sts_retrieval() -> Task:
    from seb.mteb_tasks import SNLRetrieval

    task = MTEBTask(SNLRetrieval())
    task.name = "SNL Retrieval"
    task.version = "0.0.1"
    task.domain = ["non-fiction", "wiki"]
    return task


@tasks.register("NorQuad")
def create_norquad() -> Task:
    from seb.mteb_tasks import NorQuadRetrieval

    task = MTEBTask(NorQuadRetrieval())
    task.name = "NorQuad"
    task.version = "0.0.1"
    task.domain = ["non-fiction", "wiki"]
    task.task_subtypes = ["Question-answering"]
    return task
