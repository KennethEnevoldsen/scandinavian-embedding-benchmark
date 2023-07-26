from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel

from .model_interface import ModelMeta


class TaskResult(BaseModel):
    task_name: str
    task_description: str
    task_version: str
    time_of_run: datetime
    scores: Dict[str, Dict[str, Any]]  # {language: {"metric": value}}.
    main_score: str

    def get_main_score(self, lang: Optional[Iterable[str]] = None):
        main_scores = []
        if lang is None:
            lang = self.scores.keys()

        for l in lang:
            main_scores.append(self.scores[l][self.main_score])

        return sum(main_scores) / len(main_scores)

    @property
    def languages(self) -> List[str]:
        return list(self.scores.keys())

    def to_json(self):
        return self.model_dump()


class BenchmarkResults(BaseModel):
    model_meta: ModelMeta
    task_results: List[TaskResult]
