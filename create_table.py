"""
# !pip install datawrapper
"""
import json
from pathlib import Path
from typing import Optional
from numpy import isin

import pandas as pd
from datawrapper import Datawrapper

models = {
    # Relevant multilingual models
    "sentence-transformers/all-MiniLM-L6-v2": {"datawrapper_lang_code": ["dk"]},
    "KBLab/sentence-bert-swedish-cased": {"datawrapper_lang_code": ["se"]},
    # Relevant non-sentence encoders
    # DA
    "jonfd/electra-small-nordic": {"datawrapper_lang_code": ["se", "dk", "no"]},  # S
    "vesteinn/DanskBERT": {"datawrapper_lang_code": ["dk"]},  # M
    "chcaa/dfm-encoder-large-v1": {"datawrapper_lang_code": ["dk"]},  # L
    "NbAiLab/nb-bert-large": {"datawrapper_lang_code": ["no"]},  # L
    "ltg/norbert3-large": {"datawrapper_lang_code": ["no"]},  # L
    # "NbAiLab/nb-bert-base": {"datawrapper_lang_code":  ["no"]}, # M
    "ltg/norbert3-base": {"datawrapper_lang_code": ["no"]},  # M
    "KB/bert-base-swedish-cased": {"datawrapper_lang_code": ["se"]},  # M
    "KBLab/electra-small-swedish-cased-discriminator": {
        "datawrapper_lang_code": ["se"]
    },  # S
    # Multilingual baselines
    "xlm-roberta-base": {},
    # English
    "intfloat/e5-small": {"datawrapper_lang_code": ["us"]},
    "intfloat/e5-base": {"datawrapper_lang_code": ["us"]},
    "intfloat/e5-large": {"datawrapper_lang_code": ["us"]},
    # Multilingual
    "intfloat/multilingual-e5-small": {},
    "intfloat/multilingual-e5-base": {},
    "intfloat/multilingual-e5-large": {},
    # my model
    "KennethEnevoldsen/dfm-sentence-encoder-large-1": {"datawrapper_lang_code": ["dk"]},
}

task_meta = {
    "AngryTweetsClassification": {"pretty_name": "Angry Tweets", "lang": ["da"]},
    "DKHateClassification": {"pretty_name": "DKHate", "lang": ["da"]},
    "DalajClassification": {"pretty_name": "Superlim/Dalaj", "lang": ["da"]},
    "DanishPoliticalCommentsClassification": {
        "pretty_name": "Da Political Comments",
        "lang": ["da"],
    },
    "LccSentimentClassification": {"pretty_name": "LCC", "lang": ["da"]},
    "MassiveIntentClassification": {"pretty_name": "Massive Intent", "lang": ["da"]},
    "MassiveScenarioClassification": {"pretty_name": "Massive Scenario", "lang": ["da"]},
    "NoRecClassification": {"pretty_name": "NoRec", "lang": ["da"]},
    "NordicLangClassification": {
        "pretty_name": "Language Identification",
        "lang": ["da"],
    },
    "NorwegianParliament": {"pretty_name": "Norwegian Parliament", "lang": ["no"]},
    "ScalaDaClassification": {"pretty_name": "Scala-Da", "lang": ["da"]},
    "ScalaNbClassification": {"pretty_name": "Scala-Nb", "lang": ["nb"]},
    "ScalaSvClassification": {"pretty_name": "Scala-Sv", "lang": ["sv"]},
    "SwerecClassification": {"pretty_name": "SweRec", "lang": ["sv"]},
}


def load_resuls(json_file: Path):
    with open(json_file) as f:
        result = json.load(f)

    result["model"] = json_file.parent.name
    return result


def get_meta(scores: dict, lang: Optional[str]):
    # find split:
    if "test" in scores:
        split = "test"
    elif "validation" in scores:
        split = "validation"
    elif "train" in scores:
        split = "train"
    else:
        raise ValueError(f"Could not find split in {scores}")
    
    # Check if dataset has language codes
    key1 = list(scores[split].keys())[0]
    has_lang_codes = isinstance(scores[split][key1], dict)

    if has_lang_codes:
        
    # if "test" in result:
    #     main_score = result["test"].get("da", result["test"])["main_score"]
    # elif "validation" in result:
    #     main_score = result["validation"].get("da", result["validation"])["main_score"]
    # elif "train" in result:
    #     main_score = result["train"].get("da", result["train"])["main_score"]
    # else:
    #     raise ValueError(f"No main score found for {result}")


def convert_to_row(result):
    # crete markdown entrance from model
    hf_name = result["model"].replace("__", "/")
    if "datawrapper_lang_code" in models[hf_name]:
        codes = models[hf_name]["datawrapper_lang_code"]
        lang_code = " " + " ".join([f":{c}:" for c in codes])
    else:
        lang_code = ""
    hf_link = f"https://huggingface.co/{hf_name}"
    model_name_no_org = hf_name.split("/")[-1]
    md_name = f"[{model_name_no_org}]({hf_link}){lang_code}"

    task = result["mteb_dataset_name"]

    return [md_name, task, main_score]


json_files = list(Path("results").rglob("*.json"))

results = [load_resuls(json_file) for json_file in json_files]
rows = [convert_to_row(result) for result in results]

df = pd.DataFrame(rows, columns=["Model", "task", "score"])

# replace name with pretty name
df["task"] = df["task"].apply(lambda x: task_to_pretty_name[x])

# pivot task to columns
df = df.pivot(index="Model", columns="task", values="score")
# create average column
df["Average"] = df.mean(axis=1)
# have the average column first
df = df[["Average"] + list(df.columns[:-1])]
# make model name the index
df = df.reset_index()

token = "6wZrkFwygz3ZS7HcIWcy8wITa2QHrL3vqtREg3JFmdHmBgXugbSpRsZ66FOcEMeo"
dw = Datawrapper(access_token=token)
assert dw.account_info()

# Create a new chart (table). Use the for the initial creation of the chart.
# otherwise use dw.add_data(table_id, data=df) to add the data

# chart_info = dw.create_chart(
#     title="Scandinavian Sentence Embedding Benchmark",
#     chart_type="tables",
#     data=df,
# )

table_id = "7Nwjx"

# Create a new chart (table)
resp = dw.add_data(table_id, data=df)
