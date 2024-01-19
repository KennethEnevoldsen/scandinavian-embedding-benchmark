<!-- This file is auto-generated -->

# Command Line Interface

Documentation for the command line interface of SEB.

## CLI

### `run`

Runs the Benchmark either on specified models or on all registered models. 
Can save the benchmark's results, but also displays them in a table similar to the official website. 

**Examples:**
To run all models on all languages and tasks: 

```{bash} seb run ``` 

To run a model on all languages and tasks: 

```{bash}
seb run -m sentence-transformers/all-MiniLM-L6-v2
``` 

To run multiple models: To run a model on all languages and tasks: 

```{bash} 
seb run -m sentence-transformers/all-MiniLM-L6-v2,sentence-transformers/all-mpnet-base-v2
```

if you only want to limit it to a subset of languages or tasks you can use the `--languages` and `--tasks` flags.

```{bash} 
# Running a model on a subset of languages 
seb run sentence-transformers/all-MiniLM-L6-v2 -o results/ -l nb,nn 
# Running a model on a subset of tasks 
seb run sentence-transformers/all-MiniLM-L6-v2 -o results/ -t DKHate,ScaLA
```



| Argument              | Type                            | Description                                                                                                                                                | Default  |
| --------------------- | ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| `--models`, `-m`      | `Optional[list[str], NoneType]` | Model names or paths. If a model is not registrered in SEB it will be loaded using SentenceTransformers. If none are specified the whole benchmark is run. | `None`   |
| `--output-path`, `-o` | `Path`                          | Directory to save all results to.                                                                                                                          | `None`   |
| `--languages`, `-l`   | `Optional[list[str], NoneType]` | What languages subsection to run the benchmark on. If left blank it will run it on all languages.                                                          | `None`   |
| `--tasks`, `-t`       | `Optional[list[str], NoneType]` | What tasks should model be run on. Default to all tasks within the specified languages.                                                                    | `None`   |
| `--ignore-cache`      | `bool`                          | Ignores caches models. Note that SEB ships with an existing cache. You can set the cache_dir using the environmental variable SEB_CACHE_DIR                | `False`  |
| `--ignore-errors`     | `bool`                          | Should errors be ignored when running a model on a benchmark task.                                                                                         | `False`  |
| `--code`, `-c`        | `Path`                          | Code to run before executing benchmark. Useful for adding custom model to registries.                                                                      | `None`   |
| `--logging-level`     | `str`                           | Logging level for the benchmark.                                                                                                                           | `'INFO'` |