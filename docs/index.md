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

    <div style="min-height:652px"><script type="text/javascript" defer src="https://datawrapper.dwcdn.net/7Nwjx/embed.js?v=15" charset="utf-8"></script><noscript><img src="https://datawrapper.dwcdn.net/7Nwjx/full.png" alt="" /></noscript></div>

=== "Danish"

    <div style="min-height:652px"><script type="text/javascript" defer src="https://datawrapper.dwcdn.net/us1YK/embed.js?v=11" charset="utf-8"></script><noscript><img src="https://datawrapper.dwcdn.net/us1YK/full.png" alt="" /></noscript></div>    

=== "Norwegian"

    <div style="min-height:652px"><script type="text/javascript" defer src="https://datawrapper.dwcdn.net/pV87q/embed.js?v=11" charset="utf-8"></script><noscript><img src="https://datawrapper.dwcdn.net/pV87q/full.png" alt="" /></noscript></div>

=== "Swedish"

    <div style="min-height:652px"><script type="text/javascript" defer src="https://datawrapper.dwcdn.net/aL23t/embed.js?v=11" charset="utf-8"></script><noscript><img src="https://datawrapper.dwcdn.net/aL23t/full.png" alt="" /></noscript></div>


## Comparison to other benchmarks

If you use this benchmark for a relative ranking of language models you should also look at [ScandEval](https://scandeval.github.io), which as opposed to this benchmark fully fine-tunes the models. It also includes structured prediction tasks such as named entity recognition. Many of the tasks in this embedding benchmark are also included in ScandEval. A notable difference between ScandEval and this benchmark is that this one does not include machine-translated tasks.

The tasks within this benchmark are also included in the [MTEB](https://huggingface.co/spaces/mteb/leaderboard) leaderboard, though the aggregation methods very slightly. MTEB is primarily an English embedding benchmark, with a few multilingual tasks and additional languages. As a part of this project, the tasks were also added to the MTEB leaderboard.



