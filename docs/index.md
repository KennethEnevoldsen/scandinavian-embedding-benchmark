---
hide:
  - navigation
  - toc
---

# Scandinavian Embedding Benchmark

This is the documentation for the Scandinavian Embedding Benchmark. This benchmark is intended to evaluate the sentence/document embeddings of large language models.

Intended uses for this benchmark:

- Evaluating document embeddings of Scandinavian language models
- Evaluating document embeddings for multilingual models on Scandinavian languages
- Allow ranking of competing Scandinavian and multilingual models using no more compute than what a consumer laptop can provide 


=== "All"

    <iframe title="Scandinavian Sentence Embedding Benchmark" aria-label="Table" id="datawrapper-chart-7Nwjx" src="https://datawrapper.dwcdn.net/7Nwjx/16/" scrolling="no" frameborder="0" style="width: 0; min-width: 100% !important; border: none;" height="695" data-external="1"></iframe><script type="text/javascript">!function(){"use strict";window.addEventListener("message",(function(a){if(void 0!==a.data["datawrapper-height"]){var e=document.querySelectorAll("iframe");for(var t in a.data["datawrapper-height"])for(var r=0;r<e.length;r++)if(e[r].contentWindow===a.source){var i=a.data["datawrapper-height"][t]+"px";e[r].style.height=i}}}))}();
    </script>   

=== "Danish"

    <iframe title="Danish Sentence Embedding Benchmark" aria-label="Table" id="datawrapper-chart-us1YK" src="https://datawrapper.dwcdn.net/us1YK/12/" scrolling="no" frameborder="0" style="width: 0; min-width: 100% !important; border: none;" height="708" data-external="1"></iframe><script type="text/javascript">!function(){"use strict";window.addEventListener("message",(function(a){if(void 0!==a.data["datawrapper-height"]){var e=document.querySelectorAll("iframe");for(var t in a.data["datawrapper-height"])for(var r=0;r<e.length;r++)if(e[r].contentWindow===a.source){var i=a.data["datawrapper-height"][t]+"px";e[r].style.height=i}}}))}();
    </script>

=== "Norwegian"

    <iframe title="Norwegian Sentence Embedding Benchmark" aria-label="Table" id="datawrapper-chart-pV87q" src="https://datawrapper.dwcdn.net/pV87q/12/" scrolling="no" frameborder="0" style="width: 0; min-width: 100% !important; border: none;" height="677" data-external="1"></iframe><script type="text/javascript">!function(){"use strict";window.addEventListener("message",(function(a){if(void 0!==a.data["datawrapper-height"]){var e=document.querySelectorAll("iframe");for(var t in a.data["datawrapper-height"])for(var r=0;r<e.length;r++)if(e[r].contentWindow===a.source){var i=a.data["datawrapper-height"][t]+"px";e[r].style.height=i}}}))}();
    </script>

=== "Swedish"

    <iframe title="Swedish Sentence Embedding Benchmark" aria-label="Table" id="datawrapper-chart-aL23t" src="https://datawrapper.dwcdn.net/aL23t/12/" scrolling="no" frameborder="0" style="width: 0; min-width: 100% !important; border: none;" height="677" data-external="1"></iframe><script type="text/javascript">!function(){"use strict";window.addEventListener("message",(function(a){if(void 0!==a.data["datawrapper-height"]){var e=document.querySelectorAll("iframe");for(var t in a.data["datawrapper-height"])for(var r=0;r<e.length;r++)if(e[r].contentWindow===a.source){var i=a.data["datawrapper-height"][t]+"px";e[r].style.height=i}}}))}();
    </script>


## Comparison to other benchmarks

If you use this benchmark for a relative ranking of language models were you plan to fine-tune the models I would recommend looking at [ScandEval](https://scandeval.github.io), which as benchmarks the model using a cross-validated fine-tuning. It also includes structured prediction tasks such as named entity recognition. Many of the tasks in this embedding benchmark are also included in ScandEval and an attempt have been made to use the same versions of the tasks. There is a few tasks (ScandiQA) which are included in ScandEval, but not in this benchmark as they are human translation of an English dataset.

The tasks within this benchmark are also included in the [MTEB](https://huggingface.co/spaces/mteb/leaderboard) leaderboard, though the aggregation methods very slightly. MTEB is primarily an English embedding benchmark, with a few multilingual tasks and additional languages. As a part of this project, the tasks were also added to the MTEB leaderboard.



