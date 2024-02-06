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


@tasks.register("Norwegian courts")
def create_norwegian_courts() -> Task:
    task = MTEBTask(NorwegianCourtsBitextMining())
    task.name = "Norwegian courts"
    task.domain = ["legal", "non-fiction"]
    task._text_columns = ["sentence1", "sentence2"]
    return task


@tasks.register("VG Clustering")
def create_vg_clustering() -> Task:
    from seb.mteb_tasks import VGClustering

    task = MTEBTask(VGClustering())
    task.name = "VG Clustering"
    task.version = "0.0.1"
    task.domain = ["non-fiction", "news"]
    task._text_columns = ["sentences"]
    return task


@tasks.register("SNL Clustering")
def create_sts_clustering() -> Task:
    from seb.mteb_tasks import SNLClustering

    task = MTEBTask(SNLClustering())
    task.name = "SNL Clustering"
    task.version = "0.0.1"
    task.domain = ["non-fiction", "wiki"]
    task._text_columns = ["sentences"]
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
    return task
