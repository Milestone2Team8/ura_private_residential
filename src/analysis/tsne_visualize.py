"""
This module to generate chart using kmeans clusters with t-SNE
"""
import logging
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_plot_tsne_clusters(df_input: pd.DataFrame, x_scaled: np.ndarray):
    """
    Generates plot for condo clusters using t-SNE

    :param df_input: dataframe to be processed
    :type df_input: pd.Dataframe
    :param x_scaled: scaled feature matrix
    :type x_scaled: np.ndarray
    """
    logger.info("Unsupervised Learning Step 4:Generating plot for condo clusters with t-SNE.")
    df_vis = df_input.copy()
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    x_embedded = tsne.fit_transform(x_scaled)

    df_vis["market_x"] = x_embedded[:, 0]
    df_vis["market_y"] = x_embedded[:, 1]

    plt.figure(figsize=(10, 6))
    plt.scatter(df_vis["market_x"], df_vis["market_y"], c=df_vis["cluster"],
        cmap="viridis", alpha=0.6)
    plt.title("Condo clusters visualized with t-SNE")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.grid(True)

    plt.savefig("./src/data/plot/kmeans_tsne.png", bbox_inches="tight")
    plt.close()
    logger.info("Unsupervised Learning Step 4:Plot for condo clusters with t-SNE saved.")
