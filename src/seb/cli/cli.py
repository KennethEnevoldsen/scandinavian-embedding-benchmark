from radicli import Radicli

cli = Radicli()


def setup_cli():
    from .run import run_benchmark_cli

    cli.run()
