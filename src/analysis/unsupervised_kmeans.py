"""
This module performs unsupervised kmeans algorithm and
generates model
"""
import logging
import pandas as pd
import folium
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
from src.utils.load_configs import load_configs_file


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def perform_kmeans(df_input : pd.DataFrame):
    """
    First generates elbow functions, with silouette score and then performs
    kmeans algorithm accoring to best cluster count

    :param df_input: dataframe to be processed
    :type df_input: pd.Dataframe
    :return: Tuple of updated DataFrame with cluster labels, scaled feature matrix
    :rtype: Tuple[pd.DataFrame, np.ndarray]
    """

    cluster_colors = {
        0: "green",
        1: "yellow",
        2: "red",
        3: "blue",
        4: "black"
    }
    logger.info("Unsupervised Learning Step 1:Generating elbow chart for kmeans.")
    df_kmeans = df_input.copy()

    df_kmeans = df_kmeans[df_kmeans["noOfUnits"] == 1]
    df_kmeans['tenure_bin'] = df_kmeans['tenure_bin'].cat.rename_categories(
        lambda x: '9999' if x == 'Freehold' else x
    )
    df_kmeans['tenure_bin'] = df_kmeans['tenure_bin'].cat.rename_categories(
        lambda x: x.replace(" yrs", "")
    )
    df_kmeans['tenure_bin'] = pd.to_numeric(df_kmeans['tenure_bin'])
    df_kmeans['marketSegment'] = LabelEncoder().fit_transform(df_kmeans['marketSegment'])
    df_kmeans['typeOfSale'] = LabelEncoder().fit_transform(df_kmeans['typeOfSale'])
    feature_columns = load_configs_file("features.yml")["kmeans_features"]
    df_kmeans = df_kmeans.dropna(subset=feature_columns)
    x = df_kmeans[feature_columns]

    x_scaled = StandardScaler().fit_transform(x)

    i = []
    max_cluster = 10
    for k in range(1, max_cluster + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(x_scaled)
        i.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_cluster + 1), i, marker='o')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Sum of Squares)")
    plt.title("Elbow Method for Determining Optimal No of Clusters")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./src/data/plot/kmeans_elbow_plot.png")
    plt.close()
    logger.info("Unsupervised Learning Step 1:Elbow chart for kmeans saved.")
    logger.info("Unsupervised Learning Step 2:Generating silhouette score " \
        "to find optimal no of clusters.")
    best_no_of_cluster = 2
    best_score = -1
    for i in range(2, max_cluster + 1):
        kmeans = KMeans(n_clusters=i, random_state=42)
        score = silhouette_score(x_scaled, kmeans.fit_predict(x_scaled))

        if score > best_score:
            best_no_of_cluster = i
            best_score = score

    logger.info("Unsupervised Learning Step 2:Best no of cluster=%d, Best silhouette score=%.4f.",
            best_no_of_cluster, best_score)
    logger.info("Unsupervised Learning Step 3:Generating Singapore map to show clusters.")
    kmeans = KMeans(n_clusters=best_no_of_cluster, random_state=42)
    df_kmeans["cluster"] = kmeans.fit_predict(x_scaled)

    m = folium.Map(location=[df_kmeans["latitude"].mean(),
        df_kmeans["longitude"].mean()], zoom_start=1)

    for _, row in df_kmeans.iterrows():
        folium.CircleMarker(
            location=(row["latitude"], row["longitude"]),
            radius=5,
            color=cluster_colors.get(row["cluster"]),
            fill=True,
            fill_opacity=0.6,
            popup=f"Project: {row['project']}, Sale Price: {row['target_price_cpi_adjusted']}, " \
                f"Cluster: {row['cluster']}"
        ).add_to(m)

    df_kmeans.to_csv("./src/data/output/kmeans_Singapore.csv", index=False)

    m.save("./src/data/plot/kmeans_Singapore.html")
    logger.info("Unsupervised Learning Step 3:Singapore map showing condo clusters saved.")
    return df_kmeans, x_scaled
