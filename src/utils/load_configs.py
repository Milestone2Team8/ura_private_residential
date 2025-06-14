"""
Loads configs files and feature set from a yaml file.
"""

from pathlib import Path

import yaml


def load_configs_file(file_name):
    """Loads configs files like feature names in yaml format."""
    configs_dir = Path(__file__).parents[1].joinpath("configs")
    configs_file = configs_dir.joinpath(file_name)

    with open(configs_file, encoding="utf-8") as file:
        configs = yaml.safe_load(file)

    return configs


def load_features(df_input, feature_set="all_features"):
    """ "Loads selected feature set from features.yml file"""
    configs = load_configs_file("features.yml")
    all_features = configs[feature_set]

    num_features = []
    cat_features = []

    for key, feature_list in all_features.items():
        if key.startswith("num_"):
            num_features.extend(feature_list)
        elif key.startswith("cat_"):
            cat_features.extend(feature_list)

    num_features = [col for col in num_features if col in df_input.columns]
    cat_features = [col for col in cat_features if col in df_input.columns]

    return num_features, cat_features
