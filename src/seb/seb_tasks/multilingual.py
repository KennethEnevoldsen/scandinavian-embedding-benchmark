from datetime import datetime

from seb.model_interface import ModelInterface
from seb.registries import tasks
from seb.result_dataclasses import TaskResult
from seb.tasks_interface import MTEBTask, Task


@tasks.register("Massive Intent")
def create_massive_intent() -> Task:
    from mteb import MassiveIntentClassification

    task = MTEBTask(MassiveIntentClassification())
    task.name = "Massive Intent"
    task.languages = ["da", "nb", "sv"]
    task.mteb_task.langs = ["da", "nb", "sv"]  # type: ignore
    return task


@tasks.register("Massive Scenario")
def create_massive_scenario() -> Task:
    from mteb import MassiveScenarioClassification

    task = MTEBTask(MassiveScenarioClassification())
    task.name = "Massive Scenario"
    task.languages = ["da", "nb", "sv"]
    task.mteb_task.langs = ["da", "nb", "sv"]  # type: ignore
    return task


@tasks.register("ScaLA")
def create_scala() -> Task:
    from mteb import (
        ScalaDaClassification,
        ScalaNbClassification,
        ScalaNnClassification,
        ScalaSvClassification,
        __version__,
    )

    class ScalaTask(Task):
        def __init__(self) -> None:
            self.mteb_tasks = {
                "da": ScalaDaClassification(),
                "nb": ScalaNbClassification(),
                "sv": ScalaSvClassification(),
                "nn": ScalaNnClassification(),
            }
            self.main_score = "accuracy"
            self.name = "ScaLA"
            self.description = "A linguistic acceptability task for Danish, Norwegian Bokmål Norwegian Nynorsk and Swedish."
            self.version = __version__
            self.reference = "https://aclanthology.org/2023.nodalida-1.20/"
            self.languages = ["da", "nb", "sv", "nn"]

        def evaluate(self, model: ModelInterface) -> TaskResult:
            scores = {}
            for lang, mteb_task in self.mteb_tasks.items():
                mteb_task.load_data()
                score = mteb_task.evaluate(model)
                scores[lang] = score

            return TaskResult(
                task_name=self.name,
                task_version=self.version,
                time_of_run=datetime.now(),
                scores=scores,
                task_description=self.description,
                main_score=self.main_score,
            )

    return ScalaTask()


@tasks.register("Language Identification")
def create_language_identification() -> Task:
    from mteb import NordicLangClassification

    task = MTEBTask(NordicLangClassification())
    task.name = "Language Identification"

    return task
