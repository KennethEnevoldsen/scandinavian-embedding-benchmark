import os
from pathlib import Path

CACHE_DIR = Path.home() / ".cache" / "seb"


def get_cache_dir() -> Path:
    """
    Get the cache directory for SEB. Can be overridden by setting the environment
    variable SEB_CACHE_DIR.
    """
    cache_dir = os.environ.get("SEB_CACHE_DIR")
    if cache_dir is not None:
        return Path(cache_dir)
    return CACHE_DIR


def name_to_path(name: str) -> str:
    """
    Convert a name to a path.
    """
    name = name.replace("/", "__").replace(" ", "_")
    return name
