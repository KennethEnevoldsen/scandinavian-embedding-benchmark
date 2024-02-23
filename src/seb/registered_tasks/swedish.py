from seb.interfaces.mteb_task import MTEBTask
from seb.interfaces.task import Task
from seb.registries import tasks


@tasks.register("SweReC")
def create_swerec() -> Task:
    from mteb import SweRecClassification

    task = MTEBTask(SweRecClassification())
    task.name = "SweReC"
    task.domain = ["reviews"]
    task.task_subtypes = ["Sentiment Classification"]
    return task


@tasks.register("DaLAJ")
def create_dalaj() -> Task:
    from mteb import DalajClassification

    task = MTEBTask(DalajClassification())
    task.name = "DaLAJ"
    task.domain = ["fiction", "non-fiction"]
    task.task_subtypes = ["Linguistic Acceptability"]
    return task


@tasks.register("SweFAQ")
def create_swefaq() -> Task:
    from seb.mteb_tasks import SweFaqRetrieval

    task = MTEBTask(SweFaqRetrieval())
    task.name = "SweFAQ"
    task.version = "0.0.1"
    task.domain = ["non-fiction", "web"]
    task._text_columns = ["question", "candidate_answer", "correct_answer"]
    task.task_subtypes = ["Question-answering"]
    return task


@tasks.register("SwednRetrieval")
def create_swedn_retrieval() -> Task:
    from seb.mteb_tasks import SwednRetrieval

    task = MTEBTask(SwednRetrieval())
    task.name = "SwednRetrieval"
    task.version = "0.0.1"
    task.domain = ["non-fiction", "news"]
    task.task_subtypes = ["Article Retrieval"]
    return task


@tasks.register("SwednClustering")
def create_swedn_clustering() -> Task:
    from seb.mteb_tasks import SwednClustering

    task = MTEBTask(SwednClustering())
    task.name = "SwednClustering"
    task.version = "0.0.1"
    task.domain = ["non-fiction", "news"]
    task._text_columns = ["sentences"]
    task.task_subtypes = ["Thematic Clustering"]
    return task
