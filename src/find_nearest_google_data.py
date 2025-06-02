"""
Module to calculate number of Google Places points of interest 
within a specified radius for each property.
"""

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INPUT_URA_PATH = Path("./src/data/output/clean_ura_data.csv")
INPUT_GOOGLE_PATH = Path("./src/data/output/clean_google_data_combined.pkl")
OUTPUT_GOOGLE_PATH = Path("./src/data/output/clean_nearest_google_data.pkl")

def count_nearest(gdf_property, gdf_google, poi_type, radius=500):
    """
    Spatially join Google POIs to properties using a radius 
    buffer and aggregate statistics.

    Args:
        gdf_property (GeoDataFrame): Properties with lat/lng points.
        gdf_google (GeoDataFrame): Google POIs with 'types', 'rating', etc.
        poi_type (str): The POI type to filter and count.
        radius (int): Buffer radius in meters.

    Returns:
        GeoDataFrame: Aggregated statistics for the given POI type.
    """
    gdf_property_proj = gdf_property.to_crs(epsg=3414).copy()
    gdf_google_proj = gdf_google.to_crs(epsg=3414)

    gdf_google_subset = gdf_google_proj[
        gdf_google_proj['types'].apply(lambda x: poi_type in x)]

    gdf_property_proj['buffer_geom'] = gdf_property_proj.geometry.buffer(radius)
    gdf_property_proj = gdf_property_proj.set_geometry('buffer_geom')
    joined = gpd.sjoin(gdf_property_proj,
                       gdf_google_subset, how='left', predicate='contains')

    agg = joined.groupby(joined.index).agg(
        **{
            f'poi_count_{poi_type}': (
                'place_id', 'count'),
            f'sum_user_ratings_{poi_type}': (
                'user_ratings_total', 'sum'),
            f'avg_rating_{poi_type}': (
                'rating', 'mean'),
            f'avg_price_level_{poi_type}': (
                'price_level', 'mean'),
            f'price_level_obs_count_{poi_type}': (
                'price_level', lambda x: x.notna().sum())
        }
    )

    result = gdf_property.join(agg)
    return result

def find_nearby_google_poi(
    poi_type_list,
    radius = 500,
    input_ura_path=INPUT_URA_PATH,
    input_google_path=INPUT_GOOGLE_PATH,
    output_google_path=OUTPUT_GOOGLE_PATH
):
    """
    Orchestrates POI aggregation for all given types and saves the result.

    Returns:
        GeoDataFrame: Pivoted POI statistics by type per property.
    """
    df_condo = pd.read_csv(input_ura_path)
    df_condo = df_condo[["street", "latitude", "longitude"]].drop_duplicates()
    gdf_property = gpd.GeoDataFrame(
        df_condo,
        geometry=gpd.points_from_xy(df_condo["longitude"], df_condo["latitude"]),
        crs="EPSG:4326"
    )
    gdf_google = pd.read_pickle(input_google_path)

    results = []

    for poi_type in poi_type_list:
        result = count_nearest(gdf_property, gdf_google, poi_type, radius = radius)
        results.append(result)

    gdf_combined = pd.concat(results, axis=1)
    gdf_combined = gdf_combined.loc[:, ~gdf_combined.columns.duplicated()]

    gdf_combined = gdf_combined.reset_index(drop=True)
    gdf_combined.to_pickle(output_google_path)

    return gdf_combined
