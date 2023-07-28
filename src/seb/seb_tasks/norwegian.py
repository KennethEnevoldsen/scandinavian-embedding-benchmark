from seb.registries import tasks
from seb.tasks_interface import MTEBTask, Task


@tasks.register("NoReC")
def create_norec() -> Task:
    from mteb import NoRecClassification

    task = MTEBTask(NoRecClassification())
    task.name = "NoReC"
    return task


@tasks.register("Norwegian parliament")
def create_norwegian_parliament() -> Task:
    from mteb import NorwegianParliamentClassification

    task = MTEBTask(NorwegianParliamentClassification())
    task.name = "Norwegian parliament"
    return task
