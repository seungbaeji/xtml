import tomllib
import time
from pathlib import Path


def get_xtml_version() -> Path:
    current_path = Path(__file__).resolve()
    while True:
        pyproject = current_path / "pyproject.toml"
        if pyproject.exists():
            break
        if current_path == current_path.parent:  # it reach to the root dir
            return None

        current_path = current_path.parent

    with pyproject.open("rb") as f:
        pyproject_data = tomllib.load(f)
    return pyproject_data["project"]["version"]
