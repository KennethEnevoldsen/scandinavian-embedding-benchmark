"""
Script for running the benchmark and pushing the results to Datawrapper.

Example:
    
    python run_benchmark.py --data-wrapper-api-token <token>
"""

import argparse
from typing import List

import numpy as np
import pandas as pd
from datawrapper import Datawrapper

import seb
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
    "sv": "se",
    "en": "us",
}


def get_main_score(task: seb.TaskResult, langs: List[str]) -> float:
    _langs = set(langs) & set(task.languages)
    return task.get_main_score(_langs) * 100


def create_mdl_name(mdl: seb.ModelMeta):
    reference = mdl.reference
    name = mdl.name

    if reference:
        mdl_name = f"[{name}]({reference})"
    else:
        mdl_name = name

    if mdl.languages:
        lang_code = " ".join(
            [
                f":{datawrapper_lang_codes[l]}:"
                for l in mdl.languages
                if l in datawrapper_lang_codes
            ]
        )
        mdl_name = f"{mdl_name} {lang_code}"

    return mdl_name


def benchmark_result_to_row(
    result: seb.BenchmarkResults, langs: List[str]
) -> pd.DataFrame:
    mdl_name = create_mdl_name(result.meta)
    # sort by task name
    task_results = result.task_results
    sorted_tasks = sorted(task_results, key=lambda t: t.task_name)
    task_names = [t.task_name for t in sorted_tasks]
    scores = [get_main_score(t, langs) for t in sorted_tasks]  # type: ignore

    df = pd.DataFrame([scores], columns=task_names, index=[mdl_name])
    df["Average"] = np.mean(scores)  # type: ignore
    return df


def convert_to_table(results: List[seb.BenchmarkResults], langs: List[str]):
    rows = [benchmark_result_to_row(result, langs) for result in results]
    df = pd.concat(rows)
    df = df.sort_values(by="Average", ascending=False)

    # ensure that the average first column
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]

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
