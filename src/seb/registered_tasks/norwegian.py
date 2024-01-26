from seb.interfaces.mteb_task import MTEBTask
from seb.interfaces.task import Task
from seb.registered_tasks.mteb_tasks import NorwegianCourtsBitextMining, NorwegianParliamentClassification
from seb.registries import tasks


@tasks.register("NoReC")
def create_norec() -> Task:
    from mteb import NoRecClassification

    task = MTEBTask(NoRecClassification())
    task.name = "NoReC"
    task.domain = ["reviews"]
    return task


@tasks.register("Norwegian parliament")
def create_norwegian_parliament() -> Task:
    task = MTEBTask(NorwegianParliamentClassification())
    task.name = "Norwegian parliament"
    task.domain = ["spoken"]
    return task


# @tasks.register("Norwegian courts")
def create_norwegian_courts() -> Task:
    task = MTEBTask(NorwegianCourtsBitextMining())
    task.name = "Norwegian courts"
    task.domain = ["legal", "non-fiction"]
    return task


@tasks.register("VGSummarizationClustering")
def create_swedn_clustering() -> Task:
    from seb.registered_tasks.mteb_tasks_clustering import VGSummarizationClustering

    task = MTEBTask(VGSummarizationClustering())
    task.name = "VGSummarizationClustering"
    task.version = "0.0.1"
    task.domain = ["non-fiction", "news"]
    return task
