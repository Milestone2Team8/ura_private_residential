"""
Cleans and combines raw POI data from multiple CSV files into a GeoDataFrame.
"""

from pathlib import Path
from shapely.geometry import Point
import pandas as pd
import geopandas as gpd

def clean_google_data(poi_type_list):
    """
    Loads and combines raw POI data from multiple CSV files based on a list of POI types.
    Removes duplicate entries based on `place_id`, converts the cleaned DataFrame into a
    GeoDataFrame, and saves it as a single pickle file.

    Args:
        poi_type_list (list of str): List of POI types (e.g., ['restaurant', 'school'])

    Returns:
        geopandas.GeoDataFrame: Cleaned and stacked POI data with geometry.
    """
    combined_df = []

    for poi_type in poi_type_list:
        input_path = Path(f"src/data/input/raw_google_data_{poi_type}.csv")
        if not input_path.exists():
            print(f"Warning: File not found for POI type '{poi_type}': {input_path}")
            continue

        df = pd.read_csv(input_path)
        df["poi_type"] = poi_type
        combined_df.append(df)

    if not combined_df:
        raise ValueError("No valid input files found.")

    df_all = pd.concat(combined_df, ignore_index=True)
    df_all = df_all.drop_duplicates(subset="place_id")

    geometry = [Point(xy) for xy in zip(df_all['lng'], df_all['lat'])]
    gdf = gpd.GeoDataFrame(df_all, geometry=geometry, crs="EPSG:4326")

    output_path = Path("src/data/output/clean_google_data_combined.pkl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_pickle(output_path)

    return gdf
