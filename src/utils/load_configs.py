"""
Loads configs files like feature names in yaml format.
"""

from pathlib import Path

import yaml


def load_configs(file_name):
    """Loads configs files like feature names in yaml format."""
    configs_dir = Path(__file__).parents[1].joinpath("configs")
    configs_file = configs_dir.joinpath(file_name)

    with open(configs_file, encoding="utf-8") as file:
        configs = yaml.safe_load(file)

    return configs
