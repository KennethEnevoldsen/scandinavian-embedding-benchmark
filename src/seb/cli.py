"""Module for a benchmark CLI."""
import argparse
import logging
from functools import partial
from pathlib import Path
from statistics import mean
from typing import Optional

import tabulate
from sentence_transformers import SentenceTransformer

import seb

BOLD = "\033[1m"
UNDERLINE = "\033[4m"
ITALIC = "\x1B[3m"
END = "\033[0m"


def pretty_print(results: seb.BenchmarkResults):
    """Pretty prints benchmark results in a table along with the average."""
    sorted_tasks = sorted(results.task_results, key=lambda t: t.task_name)
    table = []
    scores = []
    for task_or_error in sorted_tasks:
        name = task_or_error.task_name
        if isinstance(task_or_error, seb.TaskError):
            score = "NA"
        else:
            score = task_or_error.get_main_score()
            scores.append(score)
        table.append([name, score])
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


def run_benchmark(
    model_name_or_path: str,
    languages: Optional[list[str]],
) -> seb.BenchmarkResults:
    """Runs benchmark on a given model and languages."""
    meta = seb.ModelMeta(
        name=Path(model_name_or_path).stem,
    )
    model = seb.SebModel(
        meta=meta,
        loader=partial(SentenceTransformer, model_name_or_path=model_name_or_path),  # type: ignore
    )
    benchmark = seb.Benchmark(languages)
    res = benchmark.evaluate_model(model, raise_errors=False)
    return res


def main():
    """Main function of the CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name_or_path",
        help="Name of the model on HuggingFace hub, or path to the model.",
    )
    parser.add_argument(
        "languages",
        nargs="*",
        help="List of language codes to evaluate the model on.",
    )
    parser.add_argument(
        "--save_path",
        "-o",
        default="benchmark_results.json",
        help="File to store benchmark results in.",
    )

    args = parser.parse_args()
    logging.info(f"Running benchmark with {args.model_name_or_path}...")
    if not args.languages:
        args.languages = None
    results = run_benchmark(args.model_name_or_path, args.languages)
    logging.info("Saving results...")
    save_path = Path(args.save_path)
    with save_path.open("w") as save_file:
        save_file.write(results.model_dump_json())  # type: ignore
    print("\n")
    pretty_print(results)
