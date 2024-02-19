import logging
from functools import partial
from pathlib import Path
from typing import Literal, Optional

from radicli import Arg, get_list_converter

import seb
from seb.registered_models.sentence_transformer_models import SentenceTransformerWithTaskEncode, wrap_sentence_transformer
from seb.registries import get_all_models

from .cli import cli
from .import_code import import_code
from .table import convert_to_table, pretty_print_benchmark

logger = logging.getLogger(__name__)


def build_model(model_name: str) -> seb.SebModel:
    all_models = seb.models.get_all().keys()

    if model_name in seb.models:
        logger.info("Model registered in SEB. Loading from registry.")
        return seb.models.get(model_name)()

    logger.info("Model not found in SEB registry. Wrapping using setenceTransformers.")
    logger.debug(f"Models in registries include include: {all_models}")
    meta = seb.ModelMeta(
        name=Path(model_name).stem,
    )
    model = seb.SebModel(
        meta=meta,
        encoder=seb.LazyLoadEncoder(partial(wrap_sentence_transformer, model_name=model_name)),  # type: ignore
    )
    return model


def dump_results(results: list[seb.BenchmarkResults], output_path: Path):
    for result in results:
        mdl_path_name = result.meta.get_path_name()
        result.to_disk(output_path / mdl_path_name)


@cli.command(
    "run",
    models=Arg(
        "--models",
        "-m",
        help="Model names or paths."
        " If a model is not registrered in SEB it will be loaded using SentenceTransformers."
        " If none are specified the whole benchmark is run.",
        converter=get_list_converter(str, delimiter=","),
    ),
    output_path=Arg(
        "--output-path",
        "-o",
        help="Directory to save all results to.",
    ),
    languages=Arg(
        "--languages",
        "-l",
        help="What languages subsection to run the benchmark on. If left blank it will run it on all languages.",
        converter=get_list_converter(str, delimiter=","),
    ),
    tasks=Arg(
        "--tasks",
        "-t",
        help="What tasks should model be run on. Default to all tasks within the specified languages.",
        converter=get_list_converter(str, delimiter=","),
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
    models: Optional[list[str]] = None,
    output_path: Optional[Path] = None,
    tasks: Optional[list[str]] = None,
    languages: Optional[list[str]] = None,
    ignore_cache: bool = False,
    ignore_errors: bool = False,
    code_path: Optional[Path] = None,
    logging_level: Literal["DEBUG", "INFO"] = "INFO",
) -> None:
    """
    Runs the Benchmark either on specified models or on all registered models.
    Can save the benchmark's results, but also displays them in a table similar
    to the official website.

    **Examples:**

    To run all models on all languages and tasks:
    ```{bash}
    seb run
    ```

    To run a model on all languages and tasks:
    ```{bash}
    seb run -m sentence-transformers/all-MiniLM-L6-v2
    ```

    To run multiple models:
    To run a model on all languages and tasks:
    ```{bash}
    seb run -m sentence-transformers/all-MiniLM-L6-v2,sentence-transformers/all-mpnet-base-v2
    ```

    if you only want to limit it to a subset of languages or tasks you can use the `--languages` and `--tasks` flags.
    ```{bash}
    # Running a model on a subset of languages
    seb run sentence-transformers/all-MiniLM-L6-v2 -o results/ -l nb,nn

    # Running a model on a subset of tasks
    seb run sentence-transformers/all-MiniLM-L6-v2 -o results/ -t DKHate,ScaLA
    ```

    """
    logging.basicConfig(level=logging_level)

    import_code(code_path)

    benchmark = seb.Benchmark(languages, tasks=tasks)
    if models is None:  # noqa
        emb_models = get_all_models()
    else:
        emb_models = [build_model(model_name=model_name) for model_name in models]
    benchmark_results = benchmark.evaluate_models(
        emb_models,
        use_cache=not ignore_cache,
        raise_errors=not ignore_errors,
        run_model=True,
    )

    if output_path is not None:
        output_path.mkdir(exist_ok=True)
        dump_results(benchmark_results, output_path)

    highlight = [model.meta.name for model in emb_models]

    # Dummy run all models for the sake of printing the table
    current_models = {mdl.meta.name for mdl in emb_models}
    for _, create_mdl in seb.models.get_all().items():
        mdl = create_mdl()
        if mdl.meta.name not in current_models:
            emb_models.append(mdl)

    benchmark = seb.Benchmark(languages, tasks=tasks)
    benchmark_results = benchmark.evaluate_models(
        emb_models,
        use_cache=True,
        raise_errors=False,
        run_model=False,
    )

    new_highlight_names = []
    for model in benchmark_results:
        if model.meta.name in highlight:
            model.meta.name = f"NEW: {model.meta.name}"
            new_highlight_names.append(model.meta.name)
    benchmark_df = convert_to_table(benchmark_results, languages)
    pretty_print_benchmark(benchmark_df, highlight=new_highlight_names)
