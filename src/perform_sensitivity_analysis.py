"""Plot train/test MAE sensitivity charts"""

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def extract_params(label):
    """
    Extracts max_depth and min_samples_leaf from model
    label like 'RF_1: max_depth=5, min_samples_leaf=10'
    """
    depth_match = re.search(r"max_depth=(\d+)", label)
    leaf_match = re.search(r"min_samples_leaf=(\d+)", label)
    max_depth = int(depth_match.group(1)) if depth_match else None
    min_leaf = int(leaf_match.group(1)) if leaf_match else None
    return max_depth, min_leaf

def plot_sensitivity_2d(
    sensitivity_result, metric="MAE",
    fixed_param="max_depth", fixed_value=10
): # pylint: disable=too-many-locals
    """
    Plots train/test metric means and std bands vs the varying hyperparameter,
    fixing either max_depth or min_samples_leaf to a constant value.

    Args:
        sensitivity_result (dict): nested result dict with hyperparam labels
        metric (str): e.g., "MAE", "R2", etc.
        fixed_param (str): "max_depth" or "min_samples_leaf"
        fixed_value (int): value to keep fixed
    """

    train_data, test_data = [], []

    for model_label, data in sensitivity_result.items():
        max_depth, min_leaf = extract_params(model_label)

        if (fixed_param == "max_depth" and max_depth != fixed_value) or \
           (fixed_param == "min_samples_leaf" and min_leaf != fixed_value):
            continue

        varying_val = min_leaf if fixed_param == "max_depth" else max_depth

        train_scores = [fold[metric] for fold in data["train"]]
        test_scores = [fold[metric] for fold in data["test"]]

        train_data.append({
            fixed_param: fixed_value,
            "varying_param": varying_val,
            "mean": np.mean(train_scores),
            "std": np.std(train_scores),
            "low": np.mean(train_scores) - np.std(train_scores),
            "high": np.mean(train_scores) + np.std(train_scores),
        })

        test_data.append({
            fixed_param: fixed_value,
            "varying_param": varying_val,
            "mean": np.mean(test_scores),
            "std": np.std(test_scores),
            "low": np.mean(test_scores) - np.std(test_scores),
            "high": np.mean(test_scores) + np.std(test_scores),
        })

    df_train = pd.DataFrame(train_data).sort_values("varying_param")
    df_test = pd.DataFrame(test_data).sort_values("varying_param")

    plt.figure(figsize=(10, 6))

    plt.plot(df_test["varying_param"], df_test["mean"], marker='o',
             label="Test", color="tab:blue")

    plt.fill_between(df_test["varying_param"], df_test["low"],
                     df_test["high"], color="tab:blue", alpha=0.2)

    plt.plot(df_train["varying_param"], df_train["mean"], marker='o',
             label="Train", color="tab:green")

    plt.fill_between(df_train["varying_param"], df_train["low"],
                     df_train["high"], color="tab:green", alpha=0.2)

    plt.xlabel("min_samples_leaf" if fixed_param == "max_depth" else "max_depth")

    plt.ylabel(metric)

    plt.title(f"Train/Test {metric} vs "
            f"{'min_samples_leaf' if fixed_param == 'max_depth' else 'max_depth'} "
            f"(fixed {fixed_param}={fixed_value})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"./src/data/plot/sensitivity_analysis_fixed_{fixed_param}.png",
                bbox_inches="tight")

def plot_sensitivity_3d(sensitivity_results, metric = 'R2', fold_idx = -1):
    """
    Plots 3D surface of a metric (e.g., R2 or MAE) for RF models
    vs max_depth and min_samples_leaf.

    Args:
        sensitivity_results (dict): Original nested dict with model
        keys and test/train metrics.
        metric (str): Metric to plot, e.g., 'R2', 'MAE', 'RMSE'
        fold_idx (int): Which fold to extract (default -1 = last fold)

    Example:
        plot_rf_sensitivity_3d(sensitivity_results, metric='R2')
    """

    x_vals, y_vals, z_vals = [], [], []

    for k, v in sensitivity_results.items():
        match = re.search(r'max_depth=(\d+), min_samples_leaf=(\d+)', k)

        max_depth = int(match.group(1))
        min_samples_leaf = int(match.group(2))
        test_metrics = v['test'][fold_idx]
        score = test_metrics[metric]

        x_vals.append(max_depth)
        y_vals.append(min_samples_leaf)
        z_vals.append(score)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_trisurf(x_vals, y_vals, z_vals, cmap='Greens', edgecolor='grey')

    ax.set_xlabel('max_depth')
    ax.set_ylabel('min_samples_leaf')
    ax.set_zlabel(metric)
    ax.set_title(f'3D Sensitivity Plot: {metric} vs max_depth & min_samples_leaf')
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    plt.tight_layout()
    plt.savefig("./src/data/plot/sensitivity_analysis_3d.png", bbox_inches="tight")
