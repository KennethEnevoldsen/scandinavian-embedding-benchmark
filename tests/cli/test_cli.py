from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import pytest
import seb
from seb.cli import cli, run_benchmark_cli

test_dir = Path(__file__).parent


@dataclass
class BenchmarkCliTestInput:
    model: str
    score: float
    tasks: Union[list[str], None] = None
    languages: Union[list[str], None] = None
    code_path: Union[Path, None] = None
    ignore_cache: bool = False

    def to_command(self, output_path: Path) -> list[str]:
        cmd = ["", "run", "-m", self.model, "--output-path", f"{output_path}"]
        cmd += ["--languages", f"{','.join(self.languages)}"] if self.languages else []
        cmd += ["-t", f"{','.join(self.tasks)}"] if self.tasks else []
        cmd += ["--code", f"{self.code_path}"] if self.code_path else []
        return cmd


cli_command_parametrize = pytest.mark.parametrize(
    "inputs",
    [
        BenchmarkCliTestInput("sentence-transformers/all-MiniLM-L6-v2", 0.550, tasks=["DKHate"]),
        BenchmarkCliTestInput("sentence-transformers/all-MiniLM-L6-v2", 0.525, tasks=["DKHate", "ScaLA"]),
        BenchmarkCliTestInput(
            "sentence-transformers/all-MiniLM-L6-v2",
            0.50,
            tasks=["DKHate", "ScaLA"],
            languages=["sv", "nn", "nb"],
        ),
        BenchmarkCliTestInput(
            "test_model",
            np.nan,
            code_path=(test_dir / "benchmark_cli_code_inject.py"),
            tasks=["test-encode-task"],
            ignore_cache=True,
        ),
    ],
)


def is_approximately_equal(a: float, b: float) -> bool:
    """nan safe equality check"""
    if np.isnan(a) and np.isnan(b):
        return True
    return abs(a - b) < 0.001


def load_results(path: Path) -> list[seb.BenchmarkResults]:
    return [seb.BenchmarkResults.from_disk(d) for d in path.iterdir() if d.is_dir()]


@cli_command_parametrize
def test_run_benchmark_cli(inputs: BenchmarkCliTestInput, tmp_path: Path):
    run_benchmark_cli(
        models=[inputs.model],
        languages=inputs.languages,
        tasks=inputs.tasks,
        output_path=tmp_path,
        code_path=inputs.code_path,
        ignore_cache=inputs.ignore_cache,
    )
    res = load_results(tmp_path)
    assert len(res) == 1
    bench_res = res[0]
    bench_res.task_results = [tr for tr in bench_res.task_results if tr.task_name != "test-encode-task"]
    assert is_approximately_equal(bench_res.get_main_score(), inputs.score)


@cli_command_parametrize
def test_run_cli(inputs: BenchmarkCliTestInput, tmp_path: Path):
    cli.run(inputs.to_command(tmp_path))
    res = load_results(tmp_path)

    assert len(res) == 1
    bench_res = res[0]
    bench_res.task_results = [tr for tr in bench_res.task_results if tr.task_name != "test-encode-task"]
    assert is_approximately_equal(bench_res.get_main_score(), inputs.score)
