# API


## General
General function for dealing with tasks and models implemented in SEB.

:::seb.get_task

:::seb.get_all_tasks

:::seb.get_model

:::seb.get_all_models


## Benchmark

:::seb.Benchmark

## Interfaces

SEB implements to main interfaces. A task interface which is a tasks within the Benchmark and a model interface which is a model applied to the tasks.


### Model Interface

::: seb.Encoder

::: seb.LazyLoadEncoder

::: seb.SebModel

### Task Interface

::: seb.Task

## Data Classes

SEB uses data classes to store the results of a benchmark. The following classes are available:

::: seb.BenchmarkResults

::: seb.TaskResult

::: seb.TaskError
