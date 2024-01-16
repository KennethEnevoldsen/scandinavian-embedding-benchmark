import logging
from functools import partial
from pathlib import Path
from typing import Literal, Optional

from radicli import Arg, get_list_converter
from sentence_transformers import SentenceTransformer

import seb

from .cli import cli
from .import_code import import_code
from .table import display_model_table

logger = logging.getLogger(__name__)


def build_model(model_name: str) -> seb.EmbeddingModel:
    all_models = seb.models.get_all().keys()

    if model_name in seb.models:
        logger.info("Model registered in SEB. Loading from registry.")
        return seb.models.get(model_name)()

    logger.info("Model not found in SEB registry. Wrapping using setenceTransformers.")
    logger.debug(f"Models in registries include include: {all_models}")
    meta = seb.ModelMeta(
        name=Path(model_name).stem,
    )
    model = seb.EmbeddingModel(
        meta=meta,
        loader=partial(SentenceTransformer, model_name_or_path=model_name),  # type: ignore
    )
    return model


@cli.command(
    "run",
    model_name=Arg(
        help="The model name or path. If the model is not registrered in SEB it will be loaded using SentenceTransformers."
    ),
    output_path=Arg(
        "--output-path",
        "-o",
        help="The path to save the output to. Can be a directory.",
    ),
    languages=Arg(
        "--languages",
        "-l",
        help="What languages subsection to run the benchmark on. If left blank it will run it on all languages.",
        converter=get_list_converter(str, delimiter=" "),
    ),
    tasks=Arg(
        "--tasks",
        "-t",
        help="What tasks should model be run on. Default to all tasks within the specified languages.",
        converter=get_list_converter(str, delimiter=" "),
    ),
    ignore_cache=Arg(
        "--ignore-cache",
        help="Ignores caches models. Note that SEB ships with an existing cache. You can set the cache_dir using the environmental variable SEB_CACHE_DIR",
    ),
    ignore_errors=Arg(
        "--ignore-errors",
        help="Should errors be ignored when running a model on a benchmark task.",
    ),
    code_path=Arg(
        "--code",
        "-c",
        help="Code to run before executing benchmark. Useful for adding custom model to registries.",
    ),
    logging_level=Arg("--logging-level", help="Logging level for the benchmark."),
)
def run_benchmark_cli(
    model_name: str,
    output_path: Path,
    tasks: Optional[list[str]] = None,
    languages: Optional[list[str]] = None,
    ignore_cache: bool = False,
    ignore_errors: bool = False,
    code_path: Optional[Path] = None,
    logging_level: Literal["DEBUG", "INFO"] = "INFO",
) -> None:
    """
    Runs the Benchmark on a specified model.


    **Examples:**

    To run a model on all languages and tasks

    ```{bash}
    seb run sentence-transformers/all-MiniLM-L6-v2 -o results.json
    ```

    if you only want to limit it to a subset of languages or tasks you can use the `--languages` and `--tasks` flags.
    ```{bash}
    # Running a model on a subset of languages
    seb run sentence-transformers/all-MiniLM-L6-v2 -o results.json -l nb nn

    # Running a model on a subset of tasks
    seb run sentence-transformers/all-MiniLM-L6-v2 -o results.json -t DKHate ScaLA
    ```

    """
    logging.basicConfig(level=logging_level)

    import_code(code_path)

    model = build_model(model_name=model_name)
    benchmark = seb.Benchmark(languages, tasks=tasks)
    benchmark_result = benchmark.evaluate_model(
        model,
        use_cache=not ignore_cache,
        raise_errors=not ignore_errors,
    )
    benchmark_result.to_disk(output_path)
    display_model_table(benchmark_result, languages)
