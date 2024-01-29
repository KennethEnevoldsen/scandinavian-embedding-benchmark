from typing import Any

from mteb.abstasks import AbsTaskClassification


class NorwegianParliamentClassification(AbsTaskClassification):
    # this changes the description of the tasks but otherwise is the same as the task in the MTEB benchmark
    # once we have collected a few MTEB tasks not in the MTEB benchmark we can add them back to the benchmark.
    @property
    def description(self) -> dict[str, Any]:
        return {
            "name": "NorwegianParliament",
            "hf_hub_name": "NbAiLab/norwegian_parliament",
            "description": "Norwegian parliament speeches annotated with the party of the speaker (`Sosialistisk Venstreparti` vs `Fremskrittspartiet`)",
            "reference": "https://huggingface.co/datasets/NbAiLab/norwegian_parliament",
            "type": "Classification",
            "category": "s2s",
            "eval_splits": ["test", "validation"],
            "eval_langs": ["nb"],  # assumed to be bokmål
            "main_score": "accuracy",
            "n_experiments": 10,
            "samples_per_label": 16,
            "revision": "f7393532774c66312378d30b197610b43d751972",
        }
