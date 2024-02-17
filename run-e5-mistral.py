import seb

model = seb.get_model("intfloat/e5-mistral-7b-instruct")
bench = seb.Benchmark()
bench.evaluate_model(model, raise_errors=False)

# from seb.registered_tasks.speed import CPUSpeedTask
# bench = seb.Benchmark(tasks=[CPUSpeedTask()])
# bench.evaluate_models(models, use_cache=True)
