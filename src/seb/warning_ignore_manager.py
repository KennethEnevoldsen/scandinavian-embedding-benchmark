import os


class WarningIgnoreContextManager:
    """
    A context manager for ignoring warnings. When running a task
    """

    original_states = {}  # noqa: RUF012

    def __enter__(self):
        self.ignore_tokenizers_parallelism_warning()
        self.ignore_convergence_warnings()

    def __exit__(self, exc_type, exc_value, traceback):  # noqa: ANN001
        self.restore_original_states()

    def restore_original_states(self) -> None:
        for key, value in self.original_states.items():
            if value is None:
                del os.environ[key]
            else:
                os.environ[key] = value

    def ignore_tokenizers_parallelism_warning(self) -> None:
        """
        Ignore warnings from HuggingFace tokenizers casuses by multiple threads.
        """
        tok_parallel = os.environ.get("TOKENIZERS_PARALLELISM", None)
        self.original_states["TOKENIZERS_PARALLELISM"] = tok_parallel
        if tok_parallel is None:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def ignore_convergence_warnings(self) -> None:
        """
        Ignore convergence warnings from sklearn.
        """
        p_warnings = os.environ.get("PYTHONWARNINGS", None)
        self.original_states["PYTHONWARNINGS"] = p_warnings
        os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
