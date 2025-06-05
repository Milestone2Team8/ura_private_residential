"""
This module performs unsupervised kmeans algorithm and
generates model
"""
import pandas as pd
import numpy as np
import folium
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler



def perform_kmeans(df_input : pd.DataFrame):
    """
    First copies selected features and then performs
    kmeans algorithm

    :param df_input: dataframe to be processed
    :type df_input: pd.Dataframe
    :return: updated dataframe after kmeans applied
    :rtype: pd.DataFrame
    """
    df_kmeans = df_input.copy()
    df_kmeans = df_kmeans[df_kmeans["noOfUnits"] == 1]
    X = df_kmeans[["target_price", "area" ]]
    #std_scaler = StandardScaler()
    #X_scaled = std_scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42)
    df_kmeans["cluster"] = kmeans.fit_predict(X)

    map_kmeans = [df_kmeans["latitude"].mean(), df_kmeans["longitude"].mean()]
    m = folium.Map(location=map_kmeans, zoom_start=1)

    for _, row in df_kmeans.iterrows():
        folium.CircleMarker(
            location=(row["latitude"], row["longitude"]),
            radius=5,
            color="yellow" if row["cluster"] == 0 else "green" if row["cluster"] == 1 else "red",
            fill=True,
            fill_opacity=0.6,
            popup=f"Sale Price: {row['target_price']}, Cluster: {row['cluster']}"

        ).add_to(m)
    
    df_kmeans.to_csv("./src/data/output/kmeans_Singapore.csv", index=False)

    m.save("./src/data/output/kmeans_Singapore.html")