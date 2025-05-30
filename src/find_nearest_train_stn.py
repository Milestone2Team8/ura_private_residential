"""
Module to calculate the distance of the nearest MRT or LRT train station to
each property.
"""

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INPUT_URA_PATH = Path("./src/data/output/clean_ura_data.csv")
INPUT_TRAIN_STN_PATH = Path("./src/data/output/onemap_train_stn_lat_long.csv")
OUTPUT_MRT_STN_PATH = Path("./src/data/output/clean_nearest_mrt.csv")
OUTPUT_LRT_STN_PATH = Path("./src/data/output/clean_nearest_lrt.csv")


def clean_lat_long(
    df_lat_long,
    columns,
    org_search_val_rename,
    latitude_rename,
    longitude_rename,
):
    """
    Renames the default columns and drops rows where search values are
    "NOT FOUND" from OneMAP API.

    :param df_lat_long: DataFrame containing the latitude and longitude
    :type df_lat_long: pd.DataFrame
    :param columns: Relevant columns
    :type columns: list[str]
    :param org_search_val_rename: Original search values column name
    :type org_search_val_rename: str
    :param latitude_rename: Column name for latitude
    :type latitude_rename: str
    :param longitude_rename: Column name for longitude
    :type longitude_rename: str
    :return: DataFrame containing the renamed columns
    :rtype: pd.DataFrame
    """
    df_lat_long = df_lat_long[columns]

    df_lat_long = df_lat_long.rename(
        columns={
            df_lat_long.columns[0]: org_search_val_rename.replace(" ", "_"),
            df_lat_long.columns[1]: latitude_rename.replace(" ", "_"),
            df_lat_long.columns[2]: longitude_rename.replace(" ", "_"),
        }
    )

    df_lat_long = df_lat_long[df_lat_long.iloc[:, 1] != "NOT_FOUND"]

    df_lat_long = df_lat_long.reset_index(drop=True)

    return df_lat_long


def clean_train_stn_lat_long(df_train_stn_lat_long):
    """
    Cleans the train stations latitude and longitude data results returned by OneMAP
    API. Further splits train stations into Mass Rapid Transit (MRTs) and Light
    Rail Transit (LRT) stations.

    :param df_train_stn_lat_long: DataFrame containing the latitude and longitude
                               of train stations
    :type df_train_stn_lat_long: pd.DataFrame
    :return: Two DataFrames (MRT and LRT)
    :rtype: pd.DataFrame, pd.DataFrame
    """
    mrt_mask = df_train_stn_lat_long["searchval"].str.contains("MRT")
    df_clean_mrt = df_train_stn_lat_long[mrt_mask]
    df_clean_lrt = df_train_stn_lat_long[~mrt_mask]

    df_clean_mrt = clean_lat_long(
        df_clean_mrt,
        columns=["searchval", "latitude", "longitude"],
        org_search_val_rename="mrt stations",
        latitude_rename="mrt latitude",
        longitude_rename="mrt longitude",
    )

    df_clean_lrt = clean_lat_long(
        df_clean_lrt,
        columns=["searchval", "latitude", "longitude"],
        org_search_val_rename="lrt stations",
        latitude_rename="lrt latitude",
        longitude_rename="lrt longitude",
    )

    # Keep only unique lat long to compute distance to amenity
    df_clean_mrt = df_clean_mrt.drop_duplicates(
        subset=["mrt_latitude", "mrt_longitude"]
    )
    df_clean_lrt = df_clean_lrt.drop_duplicates(
        subset=["lrt_latitude", "lrt_longitude"]
    )

    return df_clean_mrt, df_clean_lrt


def find_nearest(df_property, df_amenity, df_property_cols, amenity_cols):
    """
    Finds the nearest amenity (e.g., MRT or LRT station) for each property using
    geospatial joins.

    :param df_property: DataFrame containing property locations
    :type df_property: pd.DataFrame
    :param df_amenity: DataFrame containing amenity locations (e.g., MRT stations)
    :type df_amenity: pd.DataFrame
    :param df_property_cols: Column names for property longitude and latitude
    :type df_property_cols: list[str]
    :param amenity_cols: Column names for amenity longitude and latitude
    :type amenity_cols: list[str]
    :return: DataFrame with nearest amenities and distances (in meters) for each property
    :rtype: pd.DataFrame
    """
    # Convert property and amenity DataFrames to GeoDataFrames
    gdf_property = gpd.GeoDataFrame(
        df_property,
        geometry=gpd.points_from_xy(
            df_property[df_property_cols[0]], df_property[df_property_cols[1]]
        ),
        crs="EPSG:4326",
    )
    gdf_amenity = gpd.GeoDataFrame(
        df_amenity,
        geometry=gpd.points_from_xy(
            df_amenity[amenity_cols[0]], df_amenity[amenity_cols[1]]
        ),
        crs="EPSG:4326",
    )

    # Convert to a metric projection (meters) for accurate distance and buffering
    gdf_property = gdf_property.to_crs(epsg=3414)  # Singapore SVY21
    gdf_amenity = gdf_amenity.to_crs(epsg=3414)

    # Find nearest amenity
    df_nearest = gpd.sjoin_nearest(
        gdf_property,
        gdf_amenity,
        how="left",
        distance_col="nearest_distance_m",
    )

    df_nearest = df_nearest.drop(columns=["geometry", "index_right"])

    return df_nearest


def find_nearest_train_stn(
    input_ura_path=INPUT_URA_PATH,
    input_train_stn_path=INPUT_TRAIN_STN_PATH,
    output_mrt_stn_path=OUTPUT_MRT_STN_PATH,
    output_lrt_stn_path=OUTPUT_LRT_STN_PATH,
):
    """
    Processes URA and train station data to find the nearest MRT and LRT stations
    for each condominium.

    :param input_ura_path: File path to the URA condominium data
    :type input_ura_path: str
    :param input_train_stn_path: File path to the train stations data
    :type input_train_stn_path: str
    :param output_mrt_stn_path: File path to write the nearest MRT station results
    :type output_mrt_stn_path: str
    :param output_lrt_stn_path: File path to write the nearest LRT station results
    :type output_lrt_stn_path: str
    :return: DataFrames containing the nearest MRT and LRT stations for each condominium
    :rtype: pd.DataFrame, pd.DataFrame
    """
    df_condo = pd.read_csv(input_ura_path)
    df_train_stn = pd.read_csv(input_train_stn_path)

    df_condo = df_condo[["street", "latitude", "longitude"]]
    df_condo = df_condo.drop_duplicates(subset=["latitude", "longitude"])
    df_mrt, df_lrt = clean_train_stn_lat_long(df_train_stn)

    df_nearest_mrt = find_nearest(
        df_condo,
        df_mrt,
        ["longitude", "latitude"],
        ["mrt_longitude", "mrt_latitude"],
    )
    df_nearest_lrt = find_nearest(
        df_condo,
        df_lrt,
        ["longitude", "latitude"],
        ["lrt_longitude", "lrt_latitude"],
    )

    df_nearest_mrt.to_csv(output_mrt_stn_path, index=False)
    df_nearest_lrt.to_csv(output_lrt_stn_path, index=False)

    return df_nearest_mrt, df_nearest_lrt
