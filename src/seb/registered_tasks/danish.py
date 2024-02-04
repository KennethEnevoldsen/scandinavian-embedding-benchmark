from seb.interfaces.task import Task
from seb.registries import tasks

from ..interfaces.mteb_task import MTEBTask


@tasks.register("Angry Tweets")
def create_angry_tweets() -> Task:
    from mteb import AngryTweetsClassification

    task = MTEBTask(AngryTweetsClassification())
    task.name = "Angry Tweets"
    task.domain = ["social"]
    return task


@tasks.register("LCC")
def create_lcc() -> Task:
    from mteb import LccSentimentClassification

    task = MTEBTask(LccSentimentClassification())
    task.name = "LCC"  # type: ignore
    task.domain = [
        "legal",
        "web",
        "news",
        "social",
        "fiction",
        "non-fiction",
        "academic",
        "government",
    ]
    return task


@tasks.register("Bornholm Parallel")
def create_bornholm_parallel() -> Task:
    from mteb import BornholmBitextMining

    task = MTEBTask(BornholmBitextMining())
    task.name = "Bornholm Parallel"
    task.domain = ["poetry", "wiki", "fiction", "web", "social"]
    task._text_columns = ["sentence1", "sentence2"]
    return task


@tasks.register("DKHate")
def create_dkhate() -> Task:
    from mteb import DKHateClassification

    task = MTEBTask(DKHateClassification())
    task.name = "DKHate"
    task.domain = ["social"]
    return task


@tasks.register("Da Political Comments")
def create_da_political_comments() -> Task:
    from mteb import DanishPoliticalCommentsClassification

    task = MTEBTask(DanishPoliticalCommentsClassification())
    task.name = "Da Political Comments"
    task.domain = ["social"]
    task.reference = "https://huggingface.co/datasets/danish_political_comments"  # TODO: Make a PR for MTEB to add this reference
    return task


@tasks.register("DanFEVER")
def create_dan_fever() -> Task:
    from seb.mteb_tasks import DanFever

    task = MTEBTask(DanFever())
    task.name = "DanFEVER"
    task.domain = ["wiki", "non-fiction"]
    return task


@tasks.register("TV2Nord Retrieval")
def create_tv2nord_retrieval() -> Task:
    from seb.mteb_tasks import TV2Nordretrieval

    task = MTEBTask(TV2Nordretrieval())
    task.name = "TV2Nord Retrieval"
    task.domain = ["news", "non-fiction"]
    return task


@tasks.register("Twitterhjerne")
def create_twitterhjerne() -> Task:
    from seb.mteb_tasks import TwitterHjerneRetrieval

    task = MTEBTask(TwitterHjerneRetrieval())
    task.name = "Twitterhjerne"
    task.domain = ["social"]
    return task
