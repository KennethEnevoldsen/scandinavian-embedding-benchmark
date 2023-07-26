from seb.registries import tasks
from seb.tasks_interface import MTEBTask, Task


@tasks.register("Angry Tweets")
def create_angry_tweets() -> Task:
    from mteb import AngryTweetsClassification

    task = MTEBTask(AngryTweetsClassification())
    task.name = "Angry Tweets"
    return task


@tasks.register("LCC")
def create_lcc() -> Task:
    from mteb import LccSentimentClassification

    task = MTEBTask(LccSentimentClassification())
    task.name = "LCC"  # type: ignore
    return task


@tasks.register("Bornholm Parallel")
def create_bornholm_parallel() -> Task:
    from mteb import BornholmBitextMining

    task = MTEBTask(BornholmBitextMining())
    task.name = "Bornholm Parallel"
    return task


@tasks.register("DKHate")
def create_dkhate() -> Task:
    from mteb import DKHateClassification

    task = MTEBTask(DKHateClassification())
    task.name = "DKHate"
    return task


@tasks.register("Da Political Comments")
def create_da_political_comments() -> Task:
    from mteb import DanishPoliticalCommentsClassification

    task = MTEBTask(DanishPoliticalCommentsClassification())
    task.name = "Da Political Comments"
    return task
