import json
from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
from pydantic import BaseModel

from .interfaces.language import Language
from .interfaces.model import ModelMeta


class TaskResult(BaseModel):
    """
    Dataclass for storing task results.

    Attributes:
        task_name: Name of the task.
        task_description: Description of the task.
        task_version: Version of the task.
        time_of_run: Time of the run.
        scores: Dictionary of scores on the form {language: {"metric": value}}.
        main_score: Name of the main score.
    """

    task_name: str
    task_description: str
    task_version: str
    time_of_run: datetime
    scores: dict[Language, dict[str, Union[float, str]]]  # {language: {"metric": value}}.
    main_score: str

    def get_main_score(self, lang: Optional[Iterable[str]] = None) -> float:
        """
        Returns the main score for a given set of languages.

        Args:
            lang: List of languages to get the main score for.

        Returns:
            The main score.
        """
        main_scores = []
        if lang is None:
            lang = self.scores.keys()

        for l in lang:
            main_scores.append(self.scores[l][self.main_score])  # type: ignore

        return sum(main_scores) / len(main_scores)

    @property
    def languages(self) -> list[Language]:
        """
        Returns the languages of the task.
        """
        return list(self.scores.keys())

    @classmethod
    def from_disk(cls, path: Path) -> "TaskResult":
        """
        Load task results from a path.
        """
        with path.open("r") as f:
            task_results = json.load(f)
        return cls(**task_results)

    def to_disk(self, path: Path) -> None:
        """
        Write task results to a path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        json_str: str = self.model_dump_json()  # type: ignore

        with path.open("w") as f:
            f.write(json_str)

    def name_to_path(self) -> str:
        """
        Convert a name to a path.
        """
        name = self.task_name.replace("/", "__").replace(" ", "_")
        return name


class TaskError(BaseModel):
    task_name: str
    error: str
    time_of_run: datetime
    languages: list[str] = []

    def to_disk(self, path: Path) -> None:
        """
        Write task results to a path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        json_str: str = self.model_dump_json()  # type: ignore

        with path.open("w") as f:
            f.write(json_str)

    @classmethod
    def from_disk(cls, path: Path) -> "TaskError":
        """
        Load task results from a path.
        """
        with path.open() as f:
            task_results = json.load(f)
        return cls(**task_results)

    @staticmethod
    def get_main_score(lang: Optional[Iterable[str]] = None) -> float:  # noqa: ARG004
        return np.nan

    def name_to_path(self) -> str:
        """
        Convert a name to a path.
        """
        name = self.task_name.replace("/", "__").replace(" ", "_")
        return name


class BenchmarkResults(BaseModel):
    """
    Dataclass for storing benchmark results.

    Attributes:
        meta: ModelMeta object.
        task_results: List of TaskResult objects.
    """

    meta: ModelMeta
    task_results: list[Union[TaskResult, TaskError]]

    def get_main_score(self, lang: Optional[Iterable[Language]] = None) -> float:
        scores = [t.get_main_score(lang) for t in self.task_results]
        if scores:
            return sum(scores) / len(scores)
        return np.nan

    def __iter__(self) -> Iterator[Union[TaskResult, TaskError]]:  # type: ignore
        return iter(self.task_results)

    def __getitem__(self, index: int) -> Union[TaskResult, TaskError]:
        return self.task_results[index]

    def __len__(self) -> int:
        return len(self.task_results)

    def to_disk(self, path: Path) -> None:
        """
        Write task results to a path.
        """
        if path.is_file():
            raise ValueError("Can't save BenchmarkResults to a file. Path must be a directory.")
        path.mkdir(parents=True, exist_ok=True)
        for task_result in self.task_results:
            if isinstance(task_result, TaskResult):
                task_result.to_disk(path / f"{task_result.task_name}.json")
            else:
                task_result.to_disk(path / f"{task_result.task_name}.error.json")

        meta_path = path / "meta.json"
        self.meta.to_disk(meta_path)

    @classmethod
    def from_disk(cls, path: Path) -> "BenchmarkResults":
        """
        Load task results from a path.
        """
        if not path.is_dir():
            raise ValueError("Can't load BenchmarkResults from path: {path}. Path must be a directory.")
        task_results = []
        for file in path.glob("*.json"):
            if file.stem == "meta":
                continue
            if file.stem.endswith(".error"):
                task_results.append(TaskError.from_disk(file))
            else:
                task_results.append(TaskResult.from_disk(file))

        meta_path = path / "meta.json"
        meta = ModelMeta.from_disk(meta_path)
        return cls(meta=meta, task_results=task_results)
