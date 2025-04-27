import pathlib
from typing import Any

import yaml


PATH_TO_ASSETS = pathlib.Path(__file__).resolve().parent
TESTRUN: bool = False


def load_yaml(path_to_yaml: pathlib.Path) -> dict[str, Any]:
    """Load and parse a YAML file into a Python dictionary."""
    with open(path_to_yaml) as f:
        data = yaml.safe_load(f)
    return data
