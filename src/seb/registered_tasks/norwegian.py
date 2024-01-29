from seb.interfaces.mteb_task import MTEBTask
from seb.interfaces.task import Task
from seb.mteb_tasks import NorwegianCourtsBitextMining, NorwegianParliamentClassification
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


@tasks.register("VGClustering")
def create_swedn_clustering() -> Task:
    from seb.mteb_tasks import VGClustering

    task = MTEBTask(VGClustering())
    task.name = "VGClustering"
    task.version = "0.0.1"
    task.domain = ["non-fiction", "news"]
    return task
