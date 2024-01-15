from seb.registries import tasks
from seb.tasks_interface import MTEBTask, Task


@tasks.register("SweReC")
def create_swerec() -> Task:
    from mteb import SweRecClassification

    task = MTEBTask(SweRecClassification())
    task.name = "SweReC"
    task.domain = ["reviews"]
    return task


@tasks.register("DaLAJ")
def create_dalaj() -> Task:
    from mteb import DalajClassification

    task = MTEBTask(DalajClassification())
    task.name = "DaLAJ"
    task.domain = ["fiction", "non-fiction"]
    return task


@tasks.register("SweFAQ")
def create_swefaq() -> Task:
    from seb.mteb_tasks import SweFaqRetrieval

    task = MTEBTask(SweFaqRetrieval())
    task.name = "SweFAQ"
    task.version = "0.0.1"
    task.domain = ["non-fiction", "web"]
    task._text_columns = ["question", "candidate_answer", "correct_answer"]
    return task


# temporarily disabled - will be added back in the future (along with the new datasets)
# @tasks.register("Swedn")
def create_swedn() -> Task:
    from seb.mteb_tasks import SwednSummarizationSTS

    task = MTEBTask(SwednSummarizationSTS())
    task.name = "Swedn"
    task.version = "0.0.1"
    task.domain = ["non-fiction", "news"]
    return task
