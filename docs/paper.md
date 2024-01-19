# Introduction
- importance of embeddings models (search, RAG)
- few benchmarks for scandinavian languages


## Contributions:
- Creates benchmark for scandinavian languages
  - integrates with MTEB
  - with broad coverage of both domains and use-cases
- Allow for custom encoding methods dependent on task (as opposed to mteb)
- Added a series of new datasets (?)
- easily extendable

## (Design principles)
- flexible (easy to add new models)
- easy to run on even small laptops
- minimal dependencies besides MTEB
- It should be transparent how models are run as often the exact prompt used can notably influence performance. --> this models are implemented as a part of the bencmark.

# Results

- conflict between language identification and lang. alignment


- translation then embed comparison
