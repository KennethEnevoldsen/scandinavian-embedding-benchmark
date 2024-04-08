"""
Script for running the benchmark and pushing the results to Datawrapper.

Example:

    python update_benchmark_tables.py --data-wrapper-api-token <token>
"""

import argparse
from collections import defaultdict
from collections.abc import Sequence
from typing import Optional

import numpy as np
import pandas as pd
import seb
from datawrapper import Datawrapper
from seb.full_benchmark import BENCHMARKS
from seb.registered_tasks.speed import CPUSpeedTask

subset_to_chart_id = {
    "Mainland Scandinavian": "7Nwjx",
    "Danish": "us1YK",
    "Norwegian": "pV87q",
    "Swedish": "aL23t",
    "Domain": "F00q5",
    "Task Type": "4jkip",
    "Speed x Performance": "oXdUJ",
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


def create_mdl_name_w_reference(mdl: seb.ModelMeta) -> str:
    reference = mdl.reference
    name: str = mdl.name

    mdl_name = f"[{name}]({reference})" if reference else name
    lang_flag = get_flag(mdl.languages)
    mdl_name = f"{mdl_name} {lang_flag}"

    return mdl_name


def get_speed_results(model_meta: seb.ModelMeta) -> Optional[float]:
    model = seb.get_model(model_meta.name)
    TOKENS_IN_UGLY_DUCKLING = 3591

    speed_task = CPUSpeedTask()
    speed_result = seb.run_task(speed_task, model, raise_errors=False, use_cache=True, run_model=False)
    if isinstance(speed_result, seb.TaskResult):
        speed_in_seconds = speed_result.get_main_score()
        word_per_seconds = TOKENS_IN_UGLY_DUCKLING / speed_in_seconds
        return word_per_seconds
    return None


def benchmark_result_to_row(
    result: seb.BenchmarkResults,
    langs: list[str],
) -> pd.DataFrame:
    mdl_name_w_link = create_mdl_name_w_reference(result.meta)
    # sort by task name
    task_results = result.task_results
    sorted_tasks = sorted(task_results, key=lambda t: t.task_name)
    task_names = [t.task_name for t in sorted_tasks]
    scores = [get_main_score(t, langs) for t in sorted_tasks]  # type: ignore

    df = pd.DataFrame([scores], columns=task_names, index=[mdl_name_w_link])
    df["Model name"] = result.meta.name
    df["Average Score"] = result.get_main_score() * 100
    df["Open Source"] = open_source_to_string(result.meta.open_source)
    df["Embedding Size"] = result.meta.embedding_size
    df["WPS (CPU)"] = get_speed_results(result.meta)
    return df


def create_n_datasets_row_for_domains() -> pd.DataFrame:
    tasks: list[seb.Task] = seb.get_all_tasks()
    domains = sorted({d for t in tasks for d in t.domain})
    domain2tasks = {d: [t.name for t in tasks if d in t.domain] for d in domains}
    scores = []
    domain_names = []
    n_datasets = []
    for d, ts in domain2tasks.items():
        domain_names.append(d.capitalize())
        n_datasets.append(len(ts))
    return pd.DataFrame([n_datasets], columns=domain_names, index=["N. Datasets"])


def create_n_datasets_row_for_task_types() -> pd.DataFrame:
    tasks: list[seb.Task] = seb.get_all_tasks()
    task_type = sorted({t_type for t in tasks for t_type in [t.task_type, *t.task_subtypes]})
    tasktype2tasks = {tt: [t.name for t in tasks if tt == t.task_type or tt in t.task_subtypes] for tt in task_type}
    scores = []
    task_type_names = []
    n_datasets = []
    for t, ts in tasktype2tasks.items():
        task_type_names.append(t.capitalize())
        n_datasets.append(len(ts))
    return pd.DataFrame([n_datasets], columns=task_type_names, index=["N. Datasets"])


def benchmark_result_to_domain_row(
    result: seb.BenchmarkResults,
    langs: list[str],
) -> pd.DataFrame:
    tasks: list[seb.Task] = seb.get_all_tasks()
    domains = sorted({d for t in tasks for d in t.domain})
    domain2tasks = {d: [t.name for t in tasks if d in t.domain] for d in domains}

    scores = []
    domain_names = []
    n_datasets = []
    for d, ts in domain2tasks.items():
        task_results = [r for r in result.task_results if r.task_name in ts]
        _scores = np.array([get_main_score(t, langs) for t in task_results])  # type: ignore
        score = np.mean(_scores)
        scores.append(score)
        domain_names.append(d.capitalize())
        n_datasets.append(len(ts))

    mdl_name = create_mdl_name_w_reference(result.meta)
    df = pd.DataFrame([scores], columns=domain_names, index=[mdl_name])
    df["Average Score"] = result.get_main_score() * 100
    df["Open Source"] = open_source_to_string(result.meta.open_source)
    df["Embedding Size"] = result.meta.embedding_size
    df["WPS (CPU)"] = get_speed_results(result.meta)
    return df


def benchmark_result_to_task_type_row(
    result: seb.BenchmarkResults,
    langs: list[str],
) -> pd.DataFrame:
    tasks: list[seb.Task] = seb.get_all_tasks()
    task_type = sorted({t_type for t in tasks for t_type in [t.task_type, *t.task_subtypes]})
    tasktype2tasks = {tt: [t.name for t in tasks if tt == t.task_type or tt in t.task_subtypes] for tt in task_type}

    scores = []
    task_type_names = []
    n_datasets = []
    for t, ts in tasktype2tasks.items():
        task_results = [r for r in result.task_results if r.task_name in ts]
        _scores = np.array([get_main_score(t, langs) for t in task_results])  # type: ignore
        score = np.mean(_scores)
        scores.append(score)
        task_type_names.append(t.capitalize())
        n_datasets.append(len(ts))

    mdl_name = create_mdl_name_w_reference(result.meta)
    df = pd.DataFrame([scores], columns=task_type_names, index=[mdl_name])
    df["Average Score"] = result.get_main_score() * 100
    df["Open Source"] = open_source_to_string(result.meta.open_source)
    df["Embedding Size"] = result.meta.embedding_size
    df["WPS (CPU)"] = get_speed_results(result.meta)
    return df


def convert_to_table(
    results: list[seb.BenchmarkResults],
    langs: list[str],
) -> pd.DataFrame:
    rows = [benchmark_result_to_row(result, langs) for result in results]
    df = pd.concat(rows)
    df = df.sort_values(by="Average Score", ascending=False)
    df["Average Rank"] = compute_avg_rank(df)
    # df["Average Rank (Bootstrapped)"] = compute_avg_rank_bootstrap(df) # noqa

    # ensure that the average and open source are the first column
    cols = df.columns.tolist()
    first_columns = ["Average Score", "Average Rank", "Open Source", "Embedding Size", "WPS (CPU)"]
    other_cols = sorted(c for c in cols if c not in first_columns)
    df = df[first_columns + other_cols]

    # convert name to column
    df = df.reset_index()
    df = df.rename(columns={"index": "Model"})
    df = df.sort_values(by="Model", ascending=True)

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
    df = df.drop(columns=["Average Score", "Open Source", "Embedding Size", "Model name", "WPS (CPU)"])

    ranks = df.rank(axis=0, ascending=False, na_option="bottom")
    avg_ranks = ranks.mean(axis=1)
    return avg_ranks


def compute_avg_rank_bootstrap(df: pd.DataFrame, n_samples: int = 100) -> pd.Series:
    """
    For all models bootstrap a set of tasks and compute the average rank. Repeat this n_samples times.
    """
    df = df.drop(columns=["Average Score", "Open Source", "Embedding Size", "Average Rank", "WPS (CPU)", "Model name"])
    tasks = np.array(df.columns.tolist())
    n_tasks = len(tasks)
    model2rank = defaultdict(list)

    for _ in range(n_samples):
        bootstrap_tasks = np.random.choice(tasks, n_tasks, replace=True)
        ranks = df[bootstrap_tasks].rank(axis=0, ascending=False, na_option="bottom")
        avg_ranks = ranks.mean(axis=1)
        for model, rank in avg_ranks.items():
            model2rank[model].append(rank)

    avg_ranks = {model: np.mean(ranks) for model, ranks in model2rank.items()}
    ci = {model: np.percentile(ranks, [2.5, 97.5]) for model, ranks in model2rank.items()}
    # create "{avg_rank} ({ci_low}-{ci_high})" string
    avg_ranks_ = {model: f"{avg_ranks[model]:.1f} [{ci_low:.1f}, {ci_high:.1f}]" for model, (ci_low, ci_high) in ci.items()}
    return pd.Series(avg_ranks_)


def create_domain_table(
    results: list[seb.BenchmarkResults],
    langs: list[str],
) -> pd.DataFrame:
    rows = [benchmark_result_to_domain_row(result, langs) for result in results]
    df = pd.concat(rows)
    df = pd.concat([df, create_n_datasets_row_for_domains()])
    df = df.sort_values(by="Average Score", ascending=False)
    cols = df.columns.tolist()
    first_columns = ["Average Score", "Open Source", "Embedding Size", "WPS (CPU)"]
    other_cols = sorted(c for c in cols if c not in first_columns)
    df = df[first_columns + other_cols]

    # convert name to column
    df = df.reset_index()
    df = df.rename(columns={"index": "Model"})
    df = df.sort_values(by="Model", ascending=True)
    return df


def create_task_type_table(
    results: list[seb.BenchmarkResults],
    langs: list[str],
) -> pd.DataFrame:
    rows = [benchmark_result_to_task_type_row(result, langs) for result in results]
    df = pd.concat(rows)
    df = pd.concat([df, create_n_datasets_row_for_task_types()])
    df = df.sort_values(by="Average Score", ascending=False)
    cols = df.columns.tolist()
    first_columns = ["Average Score", "Open Source", "Embedding Size", "WPS (CPU)"]
    other_cols = sorted(c for c in cols if c not in first_columns)
    df = df[first_columns + other_cols]

    # convert name to column
    df = df.reset_index()
    df = df.rename(columns={"index": "Model"})
    df = df.sort_values(by="Model", ascending=True)
    return df


def main(data_wrapper_api_token: str):
    results = seb.run_benchmark(use_cache=True, run_models=False, raise_errors=False)

    for subset, result in results.items():
        langs = BENCHMARKS[subset]

        raw_table = convert_to_table(result, langs)
        table = raw_table.drop(columns=["Model name"])
        chart_id = subset_to_chart_id[subset]
        push_to_datawrapper(table, chart_id, data_wrapper_api_token)

        if subset == "Mainland Scandinavian":
            # Update the chart for speed x performance
            chart_id = subset_to_chart_id["Speed x Performance"]
            _table = raw_table.drop(columns=["Model"]).rename(columns={"Model name": "Model"})
            push_to_datawrapper(_table, chart_id, data_wrapper_api_token)

            # also create the summary charts for task types and domains
            table = create_domain_table(result, langs)
            chart_id = subset_to_chart_id["Domain"]
            push_to_datawrapper(table, chart_id, data_wrapper_api_token)

            table = create_task_type_table(result, langs)
            chart_id = subset_to_chart_id["Task Type"]
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
