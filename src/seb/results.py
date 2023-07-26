import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from pydantic import BaseModel

from .model_interface import ModelMeta


class TaskResult(BaseModel):
    task_name: str
    task_description: str
    task_version: str
    time_of_run: datetime
    scores: Dict[str, Dict[str, float]]  # {language: {"metric": value}}.
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

    @classmethod
    def from_disk(cls, path: Path) -> "TaskResult":
        """
        Load task results from a path.
        """
        with open(path, "r") as f:
            task_results = json.load(f)
        return cls(**task_results)

    def to_disk(self, path: Path) -> None:
        """
        Write task results to a path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        json_str: str = self.model_dump_json()

        with open(path, "w") as f:
            f.write(json_str)


class BenchmarkResults(BaseModel):
    meta: ModelMeta
    task_results: List[TaskResult]

    def __iter__(self) -> Iterator[TaskResult]:
        return iter(self.task_results)

    def __getitem__(self, index: int) -> TaskResult:
        return self.task_results[index]

    def __len__(self) -> int:
        return len(self.task_results)
