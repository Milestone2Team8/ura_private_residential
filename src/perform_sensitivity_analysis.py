"""Plot train/test MAE sensitivity vs max_depth"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_PATH = Path("./src/data/plot/sensitivity_analysis.png")


def extract_max_depth(model_label):
    """
    Extract the value of max_depth from a model label string.

    :param model_label: String label of the model (e.g., "RF_1: max_depth=15, random_state=42").
    :type model_label: str

    :return: Parsed max_depth as an integer if found, otherwise None.
    :rtype: int or None
    """
    match = re.search(r"max_depth=(\d+)", model_label)
    return int(match.group(1)) if match else None


def perform_sensitivity_analysis(sensitivity_result, metric="MAE"):
    """
    Plots train and test performance metrics with confidence bands for Random Forest models.

    This function visualizes the effect of changing the max_depth hyperparameter on a chosen
    performance metric (e.g., MAE) for both train and test sets. It computes the mean and standard
    deviation of the selected metric across folds, and plots these with shaded confidence bands.
    The result is saved as a PNG file.

    :param sensitivity_result: Dictionary of fold-wise evaluation results returned from
                               run_time_series_cv(..., all_scores=True).
    :type sensitivity_result: dict
    :param metric: Evaluation metric to plot. Default is "MAE". Options include "RMSE", "MAE", or "R2".
    :type metric: str

    :return: None. The function saves the resulting plot to OUTPUT_PATH.
    :rtype: None
    """
    train_records = []
    test_records = []

    for model_label, data in sensitivity_result.items():
        depth = extract_max_depth(model_label)

        train_scores = [fold[metric] for fold in data["train"]]
        test_scores = [fold[metric] for fold in data["test"]]

        train_mean = np.mean(train_scores)
        train_std = np.std(train_scores)
        test_mean = np.mean(test_scores)
        test_std = np.std(test_scores)

        train_records.append({
            "max_depth": depth,
            "mean": train_mean,
            "low": train_mean - train_std,
            "high": train_mean + train_std,
        })

        test_records.append({
            "max_depth": depth,
            "mean": test_mean,
            "low": test_mean - test_std,
            "high": test_mean + test_std,
        })

    df_train = pd.DataFrame(train_records).sort_values("max_depth")
    df_test = pd.DataFrame(test_records).sort_values("max_depth")

    plt.figure(figsize=(10, 6))

    plt.plot(df_test["max_depth"], df_test["mean"], marker='o',
             label="Test MAE", color="tab:blue")
    plt.fill_between(df_test["max_depth"], df_test["low"], df_test["high"],
                     color="tab:blue", alpha=0.2)

    plt.plot(df_train["max_depth"], df_train["mean"], marker='o',
             label="Train MAE", color="tab:green")
    plt.fill_between(df_train["max_depth"], df_train["low"], df_train["high"],
                     color="tab:green", alpha=0.2)

    plt.xlabel("max_depth")
    plt.ylabel(metric)
    plt.title(f"Train/Test {metric} Sensitivity vs max_depth")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, bbox_inches="tight")
