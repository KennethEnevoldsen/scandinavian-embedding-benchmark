# What is this?

This repository features two different `uv.lock` files, as the benchmark has grown to include too many models for all dependencies to be covered by a single `uv.lock` file.

# How do I use this?

- To activate the legacy `uv.lock` file, use `source envs/legacy/.venv/bin/activate`.
- To activate the newer `uv.lock` file, use `source envs/modern/.venv/bin/activate`.

# Which models are supported by which envs?

## Legacy env
The legacy env supports all benchmarks made prior to November 2025. All models and benchmarks supported by the legacy env can be seen below:
- `all-MiniLM-L6-v2`
- `BAAI__bge-m3`
- `chcaa__dfm-encoder-large-v1`
- `embed-multilingual-v3.0`
- `fasttext-cc-da-300`
- `fasttext-cc-nb-300`
- `fasttext-cc-nn-300`
- `fasttext-cc-sv-300`
- `intfloat__e5-base`
- `intfloat__e5-large`
- `intfloat__e5-mistral-7b-instruct`
- `intfloat__e5-small`
- `intfloat__multilingual-e5-base`
- `intfloat__multilingual-e5-large`
- `intfloat__multilingual-e5-large-instruct`
- `intfloat__multilingual-e5-small`
- `jealk__TTC-L2V-supervised-1`
- `jealk__TTC-L2V-supervised-2`
- `jealk__TTC-L2V-unsupervised-1`
- `jinaai__jina-embedding-b-en-v1`
- `jinaai__jina-embeddings-v3`
- `jonfd__electra-small-nordic`
- `KB__bert-base-swedish-cased`
- `kb__electra-small-swedish-cased-discriminator`
- `KBLab__sentence-bert-swedish-cased`
- `KennethEnevoldsen__dfm-sentence-encoder-large-1`
- `KennethEnevoldsen__dfm-sentence-encoder-large-exp1`
- `KennethEnevoldsen__dfm-sentence-encoder-large-exp2-no-lang-align`
- `KennethEnevoldsen__dfm-sentence-encoder-medium-v1`
- `KennethEnevoldsen__dfm-sentence-encoder-small-v1`
- `KFST__XLMRoberta-en-da-sv-nb`
- `MiMe-MeMo__MeMo-BERT-03`
- `mixedbread-ai__mxbai-embed-large-v1`
- `NbAiLab__nb-bert-base`
- `NbAiLab__nb-bert-large`
- `Qwen__Qwen3-Embedding-0.6B`
- `sentence-transformers__all-MiniLM-L6-v2`
- `sentence-transformers__LaBSE`
- `sentence-transformers__paraphrase-multilingual-MiniLM-L12-v2`
- `sentence-transformers__paraphrase-multilingual-mpnet-base-v2`
- `sentence-transformers__use-cmlm-multilingual`
- `Snowflake__snowflake-arctic-embed-l-v2.0`
- `Snowflake__snowflake-arctic-embed-m-v2.0`
- `sonar-dan`
- `sonar-nno`
- `sonar-nob`
- `sonar-swe`
- `text-embedding-3-large`
- `text-embedding-3-small`
- `text-embedding-ada-002`
- `translate-e5-large`
- `vesteinn__DanskBERT`
- `voyage-multilingual-2`
- `xlm-roberta-base`
- `xlm-roberta-large`

## Modern env
The modern env supports all benchmarks made after November 2025. All models and benchmarks supported by the modern env can be seen below:
- `Qwen3-Embedding-0.6B`
- `Qwen3-Embedding-4B`
- `Qwen3-Embedding-8B`
- `embeddinggemma-300m`
