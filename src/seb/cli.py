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
    use_cache: bool = True,
    raise_errors: bool = True,
    cache_dir: Optional[str] = None,
) -> seb.BenchmarkResults:
    """Runs benchmark on a given model and languages."""
    meta = seb.ModelMeta(
        name=Path(model_name_or_path).stem,
    )
    model = seb.EmbeddingModel(
        meta=meta,
        loader=partial(SentenceTransformer, model_name_or_path=model_name_or_path),  # type: ignore
    )
    benchmark = seb.Benchmark(languages)

    cache_dir_path = Path(cache_dir) if cache_dir else None
    res = benchmark.evaluate_model(
        model, use_cache, raise_errors, cache_dir=cache_dir_path
    )
    return res


def main():
    """Main function of the CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name_or_path",
        help="Name of the model on HuggingFace hub, or path to the model.",
    )
    parser.add_argument(
        "--languages",
        "-l",
        nargs="*",
        help="List of language codes to evaluate the model on.",
    )
    parser.add_argument(
        "--output_path",
        "-o",
        default="benchmark_results.json",
        help="File to store benchmark results in.",
    )
    parser.add_argument(
        "--ignore_cache",
        action="store_true",
        default=False,
        help="Ignore cached results.",
    )
    parser.add_argument(
        "--ignore_errors",
        action="store_true",
        default=False,
        help="Ignore errors on specific tasks during evaluation.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="Directory to store cached results in.",
    )

    args = parser.parse_args()
    logging.info(f"Running benchmark with {args.model_name_or_path}...")
    if not args.languages:
        args.languages = None

    results = run_benchmark(
        args.model_name_or_path,
        args.languages,
        use_cache=not args.ignore_cache,
        raise_errors=not args.ignore_errors,
        cache_dir=args.cache_dir,
    )
    logging.info("Saving results...")
    save_path = Path(args.save_path)
    with save_path.open("w") as save_file:
        save_file.write(results.model_dump_json())  # type: ignore
    print("\n")
    pretty_print(results)
