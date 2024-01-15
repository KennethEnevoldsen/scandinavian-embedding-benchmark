"""
Script for running the benchmark and pushing the results to Datawrapper.

Example:

    python run_benchmark.py --data-wrapper-api-token <token>
"""
import argparse
from collections.abc import Sequence

import numpy as np
import pandas as pd
import seb
from datawrapper import Datawrapper
from seb.full_benchmark import BENCHMARKS

subset_to_chart_id = {
    "Mainland Scandinavian": "7Nwjx",
    "Danish": "us1YK",
    "Norwegian": "pV87q",
    "Swedish": "aL23t",
}

datawrapper_lang_codes = {
    "da": "dk",
    "nb": "no",
    "nn": "no",
    "sv": "se",
    "en": "us",
}


def get_main_score(task: seb.TaskResult, langs: list[str]) -> float:
    _langs = set(langs) & set(task.languages)
    return task.get_main_score(_langs) * 100


def get_flag(languages: Sequence[str]) -> str:
    if languages:
        flags = []
        for l in languages:
            if l in datawrapper_lang_codes:
                flags.append(datawrapper_lang_codes[l])
        flags = list(set(flags))  # remove duplicates (e.g. nb and nn)
        return " ".join([f":{f}:" for f in flags])
    return "🌐"


def open_source_to_string(open_source: bool) -> str:
    return "✓" if open_source else "✗"


def create_mdl_name(mdl: seb.ModelMeta) -> str:
    reference = mdl.reference
    name: str = mdl.name

    mdl_name = f"[{name}]({reference})" if reference else name
    lang_flag = get_flag(mdl.languages)
    mdl_name = f"{mdl_name} {lang_flag}"

    return mdl_name


def benchmark_result_to_row(
    result: seb.BenchmarkResults,
    langs: list[str],
) -> pd.DataFrame:
    mdl_name = create_mdl_name(result.meta)
    # sort by task name
    task_results = result.task_results
    sorted_tasks = sorted(task_results, key=lambda t: t.task_name)
    task_names = [t.task_name for t in sorted_tasks]
    scores = [get_main_score(t, langs) for t in sorted_tasks]  # type: ignore

    df = pd.DataFrame([scores], columns=task_names, index=[mdl_name])
    df["Average Score"] = result.get_mean_score()  # type: ignore
    df["Open Source"] = open_source_to_string(result.meta.open_source)
    df["Embedding Size"] = result.meta.embedding_size
    return df


def convert_to_table(
    results: list[seb.BenchmarkResults],
    langs: list[str],
) -> pd.DataFrame:
    rows = [benchmark_result_to_row(result, langs) for result in results]
    df = pd.concat(rows)
    df = df.sort_values(by="Average Score", ascending=False)
    df["Average Rank"] = compute_avg_rank(df)

    # ensure that the average and open source are the first column
    cols = df.columns.tolist()
    first_columns = ["Average Score", "Average Rank", "Open Source", "Embedding Size"]
    other_cols = sorted(c for c in cols if c not in first_columns)
    df = df[first_columns + other_cols]

    # convert name to column
    df = df.reset_index()
    df = df.rename(columns={"index": "Model"})

    return df


def push_to_datawrapper(df: pd.DataFrame, chart_id: str, token: str):
    dw = Datawrapper(access_token=token)
    assert dw.account_info(), "Could not connect to Datawrapper"
    resp = dw.add_data(chart_id, data=df)
    assert 200 <= resp.status_code < 300, "Could not add data to Datawrapper"
    iframe_html = dw.publish_chart(chart_id)
    assert iframe_html, "Could not publish chart"


def compute_avg_rank(df: pd.DataFrame) -> pd.Series:
    """
    For each model in the dataset, for each task, compute the rank of the model and then compute the average rank.
    """
    df = df.drop(columns=["Average Score", "Open Source", "Embedding Size"])

    ranks = df.rank(axis=0, ascending=False)
    avg_ranks = ranks.mean(axis=1)
    return avg_ranks


def main(data_wrapper_api_token: str):
    results = seb.run_benchmark(use_cache=True)

    for subset, result in results.items():
        langs = BENCHMARKS[subset]

        table = convert_to_table(result, langs)
        chart_id = subset_to_chart_id[subset]
        push_to_datawrapper(table, chart_id, data_wrapper_api_token)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-wrapper-api-token",
        type=str,
        required=True,
        help="Datawrapper API token",
    )

    args = parser.parse_args()
    main(args.data_wrapper_api_token)
