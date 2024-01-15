from dataclasses import dataclass
from pathlib import Path

import pytest
import seb
from seb.cli import cli, run_benchmark_cli

test_dir = Path(__file__).parent


@dataclass
class BenchmarkCliTestInput:
    model_name: str
    mean_score: float
    tasks: list[str] | None = None
    languages: list[str] | None = None
    code_path: Path | None = None
    ignore_cache: bool = False

    def to_command(self, output_path: Path) -> list[str]:
        cmd = ["", "run", f"{self.model_name}", "--output-path", f"{output_path}"]
        cmd += ["--languages", f"{' '.join(self.languages)}"] if self.languages else []
        cmd += ["-t", f"{' '.join(self.tasks)}"] if self.tasks else []
        cmd += ["--code", f"{self.code_path}"] if self.code_path else []
        return cmd


cli_command_parametrize = pytest.mark.parametrize(
    "inputs",
    [
        BenchmarkCliTestInput(
            "sentence-transformers/all-MiniLM-L6-v2", 0.485, None, None
        ),
        BenchmarkCliTestInput(
            "sentence-transformers/all-MiniLM-L6-v2", 0.550, tasks=["DKHate"]
        ),
        BenchmarkCliTestInput(
            "sentence-transformers/all-MiniLM-L6-v2", 0.525, tasks=["DKHate", "ScaLA"]
        ),
        BenchmarkCliTestInput(
            "sentence-transformers/all-MiniLM-L6-v2",
            0.487,
            languages=["sv", "no", "nn"],
        ),
        BenchmarkCliTestInput(
            "sentence-transformers/all-MiniLM-L6-v2", 0.423, languages=["da"]
        ),
        BenchmarkCliTestInput(
            "test_model",
            1.0,
            code_path=test_dir / "benchmark_cli_code_inject.py",
            tasks=["test-encode-task"],
            ignore_cache=True,
        ),
    ],
)


@cli_command_parametrize
def test_run_benchmark_cli(inputs: BenchmarkCliTestInput, tmp_path: Path):
    tmp_path = tmp_path.with_suffix(".json")
    run_benchmark_cli(
        model_name=inputs.model_name,
        languages=inputs.languages,
        tasks=inputs.tasks,
        output_path=tmp_path,
        code_path=inputs.code_path,
        ignore_cache=inputs.ignore_cache,
    )
    res = seb.BenchmarkResults.from_disk(tmp_path)
    assert res.get_main_score() - inputs.mean_score < 0.001


@cli_command_parametrize
def test_run_cli(inputs: BenchmarkCliTestInput, tmp_path: Path):
    tmp_path = tmp_path.with_suffix(".json")
    cli.run(inputs.to_command(tmp_path))
    res = seb.BenchmarkResults.from_disk(tmp_path)
    assert res.get_main_score() - inputs.mean_score < 0.001
