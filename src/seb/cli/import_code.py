"""Utilities for import coding in python"""

import importlib
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Optional, Union


def import_file(name: str, loc: Union[str, Path]) -> ModuleType:
    """Import module from a file. Used to load models from a directory.

    Args:
        name: Name of module to load.
        loc: Path to the file.

    Returns:
        The loaded module.
    """
    spec = importlib.util.spec_from_file_location(name, str(loc))
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def import_code(code_path: Optional[Union[Path, str]]) -> None:
    """Helper to import Python file provided in training commands / commands
    using the config. This makes custom registered functions available.
    """
    if code_path is not None:
        if not Path(code_path).exists():
            raise ValueError(f"Path to Python code not found {Path(code_path)}")
        import_file("python_code", code_path)
