"""
Detect outliers on select features and outputs umap plots with sample data.
"""

import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.iforest import IForest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.custom_distance_metric import gower_distances
from src.utils.load_configs import load_configs
from src.utils.validate import diagnose_missing_data

RANDOM_STATE = 42
OUTPUT_PATH = Path("./src/data/output/clean_merged_outliers.csv")
OUTPUT_PLOT_PATH = Path("./src/data/plot")

CAT_IMPUTER_STRATEGY = "constant"
CAT_CONSTANT = "missing"
NUM_IMPUTER_STRATEGY = "mean"

DETECTORS = {
    "IForest": IForest(random_state=RANDOM_STATE),
    "AutoEncoder": AutoEncoder(random_state=RANDOM_STATE),
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# pylint: disable=too-many-locals, too-many-arguments, too-many-positional-arguments


def process_outliers_data(
    df_ura,
    cat_imputer_strategy=CAT_IMPUTER_STRATEGY,
    num_imputer_strategy=NUM_IMPUTER_STRATEGY,
    cat_constant=CAT_CONSTANT,
    num_constant=None,
):
    """
    Imputes missing values, encode categorical features, and scales numerical features.
    Also returns a boolean array indicating which columns are categorical.
    """
    configs = load_configs("features.yml")
    outliers_features = configs["outliers_features"]
    num_features = outliers_features["num_features"]
    cat_features = outliers_features["cat_features"]

    df_copy = df_ura[num_features + cat_features].copy()

    cat_imputer = SimpleImputer(
        strategy=cat_imputer_strategy,
        fill_value=(
            cat_constant if cat_imputer_strategy == "constant" else None
        ),
    )
    num_imputer = SimpleImputer(
        strategy=num_imputer_strategy,
        fill_value=(
            num_constant if num_imputer_strategy == "constant" else None
        ),
    )

    cat_transformer = Pipeline(
        steps=[
            ("imputer", cat_imputer),
            ("encoder", OneHotEncoder(sparse_output=False)),
        ]
    )
    num_transformer = Pipeline(
        steps=[("imputer", num_imputer), ("scaler", StandardScaler())]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            ("cat", cat_transformer, cat_features),
        ]
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

    x_processed = pipeline.fit_transform(df_copy)

    feature_names = pipeline.named_steps[
        "preprocessor"
    ].get_feature_names_out()
    is_cat = np.array(["cat__" in name for name in feature_names])

    return x_processed, is_cat


def sample_outliers_data(
    df_outliers_results,
    score_cols,
    label_cols,
    top_n_outliers,
    total_sample_size,
    random_state=RANDOM_STATE,
):
    """
    Selects the top N outliers based on score, and randomly samples the remaining data with label 0.
    """
    df_outliers_samples = {}

    for score_col, label_col in zip(score_cols, label_cols):
        df_outliers = (
            df_outliers_results[df_outliers_results[label_col] == 1]
            .sort_values(by=score_col, ascending=False)
            .head(top_n_outliers)
        )

        n_inliers = total_sample_size - top_n_outliers
        df_inliers = df_outliers_results[
            df_outliers_results[label_col] == 0
        ].sample(n=n_inliers, random_state=random_state)
        df_outliers_samples[label_col] = pd.concat(
            [df_outliers, df_inliers]
        ).reset_index(drop=True)

    return df_outliers_samples


def plot_outliers_umap(
    df_outliers_sample,
    label_col,
    gower=True,
    output_plot_path=OUTPUT_PLOT_PATH,
):
    """
    Plot a 2D UMAP of outlier samples colored by the specified label column and save the plot.
    """
    warnings.filterwarnings("ignore")

    x_processed, is_cat = process_outliers_data(df_outliers_sample)

    if gower:
        x_processed = gower_distances(x_processed, cat_features=is_cat)
        metric = "precomputed"
    else:
        metric = "euclidean"

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=50,
        min_dist=0.75,
        metric=metric,
        random_state=RANDOM_STATE,
    )
    embedding = reducer.fit_transform(x_processed)

    inliers = df_outliers_sample[label_col] == 0
    outliers = df_outliers_sample[label_col] == 1
    n_inliers = inliers.sum()
    n_outliers = outliers.sum()

    plt.figure(figsize=(10, 7))

    plt.scatter(
        embedding[inliers, 0],
        embedding[inliers, 1],
        c="blue",
        alpha=0.6,
        s=50,
        edgecolor="k",
        label=f"Sample Inliers (n={n_inliers})",
    )

    plt.scatter(
        embedding[outliers, 0],
        embedding[outliers, 1],
        c="red",
        alpha=0.6,
        s=50,
        edgecolor="k",
        label=f"Top Outliers (n={n_outliers})",
    )

    title_part = (
        label_col.split("_")[1] if len(label_col.split("_")) > 2 else label_col
    )
    plt.title(f"UMAP (2D) - Colored by {title_part} Outliers")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend()

    plt.savefig(
        output_plot_path / f"umap_{label_col}.png",
        dpi=300,
        bbox_inches="tight",
    )


def detect_outliers_generate_plots(
    df_ura, detectors=None, output_path=OUTPUT_PATH
):
    """
    Detect outliers in the provided URA property DataFrame using multiple detectors,
    generate UMAP plots, and export the DataFrame with outlier scores and labels.

    :param df_ura: Input URA property DataFrame with raw features
    :type df_ura: pd.DataFrame
    :param detectors: Dictionary of outlier detection models to apply; keys are names
                      and values are detector instances
    :type detectors: dict[str, object]
    :return: DataFrame augmented with outlier scores and labels from each detector
    :rtype: pd.DataFrame
    """
    if detectors is None:
        detectors = DETECTORS
    df_copy = df_ura.copy()
    n_detectors = len(detectors)

    logger.info("---Running Outlier Detection\n")
    x_processed, _ = process_outliers_data(df_copy)

    outlier_scores = np.zeros([x_processed.shape[0], n_detectors])
    labels = np.zeros([x_processed.shape[0], n_detectors])

    for i, (_, detector) in enumerate(detectors.items()):

        logger.info("Running Detector %d\n%s\n", i + 1, detector)
        detector.fit(x_processed)
        outlier_scores[:, i] = detector.decision_scores_
        labels[:, i] = detector.labels_

    score_cols = [
        f"outliers_{detector_name}" for detector_name in detectors.keys()
    ]
    label_cols = [
        f"outliers_{detector_name}_label" for detector_name in detectors.keys()
    ]

    outlier_scores_df = pd.DataFrame(
        np.round(outlier_scores, 3),
        columns=score_cols,
    )

    labels_df = pd.DataFrame(
        labels.astype(int),
        columns=label_cols,
    )

    df_copy = df_copy.join(outlier_scores_df)
    df_copy = df_copy.join(labels_df)

    df_outliers_samples = sample_outliers_data(
        df_copy,
        score_cols=score_cols,
        label_cols=label_cols,
        top_n_outliers=10,
        total_sample_size=150,
    )

    for label_col, df_outliers_sample in df_outliers_samples.items():
        plot_outliers_umap(df_outliers_sample, label_col)

    df_copy.to_csv(output_path, index=False)
    logger.info("Completed Outlier Detection\n")

    return df_copy
