import argparse
from typing import List

import numpy as np
import pandas as pd
from datawrapper import Datawrapper

import seb

subset_to_chart_id = {
    "all": "",
    "da": "",
    "no": "",
    "sv": "",
}

datawrapper_lang_codes = {
    "da": "dk",
    "no": "no",
    "sv": "se",
}


def create_mdl_name(mdl: seb.ModelMeta):
    reference = mdl.reference
    name = mdl.name

    if reference:
        mdl_name = f"[{name}]({reference})"
    else:
        mdl_name = name

    if mdl.languages:
        lang_code = " ".join([f":{datawrapper_lang_codes[l]}:" for l in mdl.languages])
        mdl_name = f"{mdl_name} {lang_code}"

    return mdl_name


def benchmark_result_to_row(result: seb.BenchmarkResults) -> pd.DataFrame:
    mdl_name = create_mdl_name(result.meta)
    task_names = sorted([t.task_name for t in result])
    scores = [t.get_main_score() for t in result]

    df = pd.DataFrame([scores], columns=task_names, index=[mdl_name])
    df["Average"] = np.mean(scores)  # type: ignore
    return df


def convert_to_table(results: List[seb.BenchmarkResults]):
    rows = [benchmark_result_to_row(result) for result in results]
    df = pd.concat(rows)
    df = df.sort_values(by="Average", ascending=False)
    return df


def push_to_datawrapper(df: pd.DataFrame, chart_id: str, token: str):
    dw = Datawrapper(access_token=token)
    assert dw.account_info(), "Could not connect to Datawrapper"
    resp = dw.add_data(chart_id, data=df)


def main(data_wrapper_api_token: str):
    results = seb.run_benchmark(use_cache=True)

    for subset, result in results.items():
        table = convert_to_table(result)
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
