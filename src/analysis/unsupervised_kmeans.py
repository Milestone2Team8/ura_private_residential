"""
This module performs unsupervised kmeans algorithm and
generates model
"""
import pandas as pd
import folium
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def perform_kmeans(df_input : pd.DataFrame):
    """
    First generates elbow functions, with silouette score and then performs
    kmeans algorithm accoring to best cluster count

    :param df_input: dataframe to be processed
    :type df_input: pd.Dataframe
    :return: Tuple of updated DataFrame with cluster labels and scaled feature matrix
    :rtype: Tuple[pd.DataFrame, np.ndarray]
    """

    cluster_colors = {
        0: "green",
        1: "yellow",
        2: "red"
    }
    print ("---Unsupervised Learning Step 1:Generating elbow chart for kmeans")
    df_kmeans = df_input.copy()
    df_kmeans = df_kmeans[df_kmeans["noOfUnits"] == 1]
    x = df_kmeans[["target_price", "area", "mrt_nearest_distance_m",
        "lrt_nearest_distance_m", "poi_count_restaurant", "SORA", "monthly_price_index"]]
    std_scaler = StandardScaler()
    x_scaled = std_scaler.fit_transform(x)

    i = []
    max_cluster = 8
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
    plt.savefig("./src/data/output/kmeans_elbow_plot.png")
    plt.close()
    print ("---Unsupervised Learning Step 1:Elbow chart for kmeans saved.\n")
    print ("---Unsupervised Learning Step 2:Generating silhouette score " \
        "to find optimal no of clusters.")
    best_no_of_cluster = 2
    best_score = -1
    for i in range(2, max_cluster + 1):
        kmeans = KMeans(n_clusters=i, random_state=42)
        score = silhouette_score(x_scaled, kmeans.fit_predict(x_scaled))

        if score > best_score:
            best_no_of_cluster = i
            best_score = score

    print(f"---Unsupervised Learning Step 2:Best no of cluster={best_no_of_cluster}, " \
         f"Best silhouette score={best_score:.4f} .\n")
    print("---Unsupervised Learning Step 3:Generating Singapore map to show clusters.")
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
            popup=f"Sale Price: {row['target_price']}, Cluster: {row['cluster']}"
        ).add_to(m)

    df_kmeans.to_csv("./src/data/output/kmeans_Singapore.csv", index=False)

    m.save("./src/data/output/kmeans_Singapore.html")
    print("---Unsupervised Learning Step 3:Singapore map showing condo clusters saved.\n")
    return df_kmeans, x_scaled
