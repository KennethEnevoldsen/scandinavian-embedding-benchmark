import logging
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import psutil
import torch

from seb.interfaces.language import languages_in_seb
from seb.interfaces.model import Encoder
from seb.interfaces.task import Task
from seb.result_dataclasses import TaskResult

logger = logging.getLogger(__name__)

TOKENS_IN_UGLY_DUCKLING = 3591


class CPUSpeedTask(Task):
    reference = "NA"
    version = "0.0.1"
    task_type = "Speed"
    languages = languages_in_seb
    main_score = "Inference speed (seconds)"
    domain = ["fiction"]  # noqa
    name = "Speed (CPU)"
    description = "Time taken to encode the text 'The Ugly Duckling' split by paragraphs on a CPU."
    device = "cpu"
    _dataset: Optional[list[str]] = None
    task_subtypes = []  # noqa

    def load_dataset(self) -> list[str]:
        file_path = Path(__file__).parent / "the_ugly_duckling.txt"
        with file_path.open("r") as f:
            text = f.read()
        return text.split("\n\n")

    @property
    def dataset(self) -> list[str]:
        if self._dataset is None:
            self._dataset = self.load_dataset()
        return self._dataset

    def get_documents(self) -> list[str]:
        return self.load_dataset()

    def get_time_taken(self, model: Encoder) -> float:
        dataset = self.load_dataset()
        start = time.time()
        with torch.no_grad():
            model.encode(dataset, device=self.device, task=self)
        time_taken = time.time() - start
        return time_taken

    def evaluate(self, model: Encoder) -> TaskResult:  # type: ignore
        model.encode(["encode this"])  # ensure model is loaded

        has_to_method = hasattr(model._model, "to") and isinstance(model._model.to, Callable)  # type: ignore
        if has_to_method:
            model._model = model._model.to(self.device)  # type: ignore

        run_inference = not (self.device == "cuda" and not has_to_method)
        time_taken = self.get_time_taken(model) if run_inference else np.nan

        scores: dict[str, Union[str, float]] = {
            self.main_score: time_taken,
            "words pr. second": TOKENS_IN_UGLY_DUCKLING / time_taken,
            **self.get_system_info(),
        }

        return TaskResult(
            task_name=self.name,
            task_description=self.description,
            task_version=self.version,
            scores={Language: scores for Language in self.languages},  # type: ignore
            time_of_run=datetime.now(),
            main_score=self.main_score,
        )

    def get_system_info(self) -> dict[str, str]:
        """
        Returns a dictionary with system information.
        """
        info = {}
        info["platform"] = platform.system()
        info["platform-release"] = platform.release()
        info["platform-version"] = platform.version()
        info["architecture"] = platform.machine()
        info["processor"] = platform.processor()
        info["ram"] = str(round(psutil.virtual_memory().total / (1024.0**3))) + " GB"
        info["Physical cores"] = psutil.cpu_count(logical=False)
        info["Total cores"] = psutil.cpu_count(logical=True)
        return info


class GPUSpeedTask(CPUSpeedTask):
    name = "Speed (GPU)"
    description = "Time taken to encode the text 'The Ugly Duckling' split by paragraphs on a GPU."
    device: str = "cuda"
