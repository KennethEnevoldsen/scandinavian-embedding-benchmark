"""Module for a benchmark CLI."""
import argparse
import logging
from functools import partial
from pathlib import Path
from statistics import mean

import tabulate
from sentence_transformers import SentenceTransformer

import seb


def get_main_score(task: seb.TaskResult, langs: list[str]) -> float:
    _langs = set(langs) & set(task.languages)
    return task.get_main_score(_langs) * 100


BOLD = "\033[1m"
UNDERLINE = "\033[4m"
ITALIC = "\x1B[3m"
END = "\033[0m"


def pretty_print(results: seb.BenchmarkResults, langs: list[str]):
    """Pretty prints benchmark results in a table along with the average."""
    sorted_tasks = sorted(results.task_results, key=lambda t: t.task_name)
    task_names = [t.task_name for t in sorted_tasks]
    scores = [get_main_score(t, langs) for t in sorted_tasks]  # type: ignore
    table = []
    for task, score in zip(task_names, scores):
        table.append([task, score])
    # Adding empty line before average, so it is highlighted
    table.append(["", ""])
    mean_score = str(mean(scores))
    table.append([ITALIC + "Average" + END, ITALIC + mean_score + END])
    print(
        tabulate.tabulate(
            table,
            headers=[BOLD + "Task" + END, BOLD + "Score" + END],
            tablefmt="simple",
        ),
    )


def run_benchmark(model_name_or_path: str) -> seb.BenchmarkResults:
    """Runs benchmark on a given model."""
    meta = seb.ModelMeta(
        name=Path(model_name_or_path).stem,
    )
    model = seb.SebModel(
        meta=meta,
        loader=partial(SentenceTransformer, model_name_or_path=model_name_or_path),
    )
    benchmark = seb.Benchmark()
    res = benchmark.evaluate_model(model)
    return res


def main():
    """Main function of the CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name_or_path",
        help="Name of the model on HuggingFace hub, or path to the model.",
    )
    parser.add_argument(
        "--save_path",
        default="benchmark_results.json",
        help="File to store benchmark results in.",
    )

    args = parser.parse_args()
    logging.info(f"Running benchmark with {args.model_name_or_path}...")
    results = run_benchmark(args.model_name_or_path)
    logging.info("Saving results...")
    save_path = Path(args.save_path)
    with save_path.open("w") as save_file:
        save_file.write(results.model_dump_json())
    print("Benchmark Results:")
    pretty_print(results, langs=["da", "no", "se"])
