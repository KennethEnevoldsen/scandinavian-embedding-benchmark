from typing import Optional

import pandas as pd
from rich.console import Console
from rich.table import Table

import seb
from seb.interfaces.language import Language


def get_main_score(task: seb.TaskResult, langs: Optional[list[Language]]) -> float:
    if langs is None:  # noqa
        _langs = task.languages
    else:
        _langs = set(langs) & set(task.languages)
    return task.get_main_score(_langs) * 100


def open_source_to_string(open_source: bool) -> str:
    return "✓" if open_source else "✗"


def benchmark_result_to_row(
    result: seb.BenchmarkResults,
    langs: Optional[list[str]],
) -> pd.DataFrame:
    mdl_name = result.meta.name
    # sort by task name
    task_results = result.task_results
    sorted_tasks = sorted(task_results, key=lambda t: t.task_name)
    task_names = [t.task_name for t in sorted_tasks]
    scores = [get_main_score(t, langs) for t in sorted_tasks]  # type: ignore

    df = pd.DataFrame([scores], columns=task_names, index=[mdl_name])  # type: ignore
    df["Average Score"] = result.get_main_score()  # type: ignore
    df["Open Source"] = open_source_to_string(result.meta.open_source)
    df["Embedding Size"] = result.meta.embedding_size
    return df


def convert_to_table(
    results: list[seb.BenchmarkResults],
    langs: Optional[list[str]],
) -> pd.DataFrame:
    rows = [benchmark_result_to_row(result, langs) for result in results]
    df = pd.concat(rows)
    df = df.sort_values(by="Average Score", ascending=False)
    df["Average Rank"] = compute_avg_rank(df)

    # ensure that the average and open source are the first column
    cols = df.columns.tolist()
    first_columns = [
        "Average Score",
        "Average Rank",
    ]
    other_cols = sorted(c for c in cols if c not in first_columns)
    df = df[first_columns + other_cols]
    df = df.drop(columns=["Open Source", "Embedding Size"])

    # convert name to column
    df = df.reset_index()
    df = df.rename(columns={"index": "Model"})
    ranks = [i + 1 for i in range(len(df.index))]
    df.insert(0, "Rank", ranks)  # type: ignore
    return df


def compute_avg_rank(df: pd.DataFrame) -> pd.Series:
    """
    For each model in the dataset, for each task, compute the rank of the model
    and then compute the average rank.
    """
    df = df.drop(columns=["Average Score", "Open Source", "Embedding Size"])
    ranks = df.rank(axis=0, ascending=False)
    avg_ranks: pd.Series = ranks.mean(axis=1)  # type: ignore
    return avg_ranks


MIN_WIDTHS = {
    "Rank": len("Rank"),
    "Model": len("embed-multilingual-v3.0"),
    "Average Score": len("Average"),
    "Average Rank": len("Average"),
}
NO_WRAP = {
    "Model": True,
    "Rank": True,
}


def pretty_print_benchmark(df: pd.DataFrame, highlight: list[str]):
    """Pretty prints the benchmark's results with Rich.
    If you pass a model name in highlight, the model will
    be highlighted and only the rows around it will be showed,
    otherwise the full benchmark is shown and no row is highlighted.
    """
    console = Console()
    table = Table(title="Benchmark Results")
    for column in df.columns:
        justify = "left" if column == "Model" else "right"
        table.add_column(
            column,
            justify=justify,
            overflow="ellipsis",
            min_width=MIN_WIDTHS.get(column, None),  # noqa
            no_wrap=NO_WRAP.get(column, False),
        )
    if highlight:
        models_to_display = []
        # Add top 3 models
        models_to_display.extend(df["Model"][df["Rank"] <= 3])
        # Add models surrounding the highlighted ones
        for model in highlight:
            model_rank = df[df["Model"] == model]["Rank"].iloc[0]  # type: ignore
            models_to_display.extend(df["Model"][(df["Rank"] - model_rank).abs() < 2])
    else:
        models_to_display = list(df["Model"])
    df = df[df["Model"].isin(models_to_display)]  # type: ignore
    df = df.sort_values("Rank")  # type: ignore
    for _, row in df.iterrows():
        style = "deep_sky_blue1 bold" if (row["Model"] in highlight) else None
        rank = row["Rank"]
        values = []
        for val in row:
            if isinstance(val, float):
                val = f"{val:.2f}"  # noqa
            if not isinstance(val, str):
                val = str(val)  # noqa
            values.append(val)
        table.add_row(*values, style=style)
        if (rank == 3) and highlight:
            table.add_section()
    console.print(table)
