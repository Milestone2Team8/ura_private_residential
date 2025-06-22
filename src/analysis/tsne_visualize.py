"""
This module to generate chart using kmeans clusters with t-SNE
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def sample_clusters_data(
    df_input, cluster_col="cluster", samples_per_cluster=50, random_state=42
):
    """
    Samples a fixed number of rows from each cluster in a DataFrame.

    :param df_input: Input DataFrame containing cluster assignments
    :type df_input: pd.DataFrame
    :param cluster_col: Name of the column containing cluster labels
    :type cluster_col: str
    :param samples_per_cluster: Maximum number of samples to draw from each cluster
    :type samples_per_cluster: int
    :param random_state: Random seed for reproducibility
    :type random_state: int
    :return: DataFrame containing the sampled rows from each cluster
    :rtype: pd.DataFrame
    """
    sampled_df = (
        df_input.groupby(cluster_col, group_keys=False)
        .apply(
            lambda x: x.sample(
                min(len(x), samples_per_cluster), random_state=random_state
            )
        )
        .reset_index(drop=True)
    )
    return sampled_df


def generate_plot_tsne_clusters(df_input: pd.DataFrame, x_scaled: np.ndarray):
    """
    Generates plot for condo clusters using t-SNE

    :param df_input: dataframe to be processed
    :type df_input: pd.Dataframe
    :param x_scaled: scaled feature matrix
    :type x_scaled: np.ndarray
    """
    logger.info(
        "Unsupervised Learning Step 4:Generating plot for condo clusters with t-SNE."
    )

    tnse_colors = {
        0: "green",
        1: "yellow",
        2: "red",
        3: "blue",
        4: "black",
        5: "brown",
        6: "orange",
        7: "purple",
    }

    df_vis = df_input.copy()
    # df_sampled = sample_clusters_data(df_vis, cluster_col="cluster", samples_per_cluster=100)
    # x_scaled_sampled = x_scaled[df_sampled.index]
    df_sampled = df_vis
    x_scaled_sampled = x_scaled
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    x_embedded = tsne.fit_transform(x_scaled_sampled)

    df_sampled["market_x"] = x_embedded[:, 0]
    df_sampled["market_y"] = x_embedded[:, 1]
    df_sampled["color"] = df_sampled["cluster"].map(tnse_colors)

    plt.figure(figsize=(10, 6))
    plt.scatter(
        df_sampled["market_x"],
        df_sampled["market_y"],
        c=df_sampled["color"],
        alpha=0.6,
    )
    plt.title("Condo clusters visualized with t-SNE")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.grid(True)

    plt.savefig("./src/data/plot/kmeans_tsne.png", bbox_inches="tight")
    plt.close()
    logger.info(
        "Unsupervised Learning Step 4:Plot for condo clusters with t-SNE saved."
    )
