from seb import Benchmark
import seb
benchmark = Benchmark()
model = seb.get_model("translate-e5-small")
benchmark.evaluate_model(model)