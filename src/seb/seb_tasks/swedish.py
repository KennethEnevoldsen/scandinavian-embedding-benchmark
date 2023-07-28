from seb.registries import tasks
from seb.tasks_interface import MTEBTask, Task


@tasks.register("SweReC")
def create_swerec() -> Task:
    from mteb import SweRecClassification

    task = MTEBTask(SweRecClassification())
    task.name = "SweReC"
    return task


@tasks.register("DaLAJ")
def create_dalaj() -> Task:
    from mteb import DalajClassification

    task = MTEBTask(DalajClassification())
    task.name = "DaLAJ"
    return task
