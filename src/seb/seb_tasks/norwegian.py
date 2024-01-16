from seb.mteb_tasks import (NorwegianCourtsBitextMining,
                            NorwegianParliamentClassification)
from seb.registries import tasks
from seb.tasks_interface import MTEBTask, Task


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
    task.domain = ["legal"]
    return task
