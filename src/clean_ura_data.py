"""
Module to clean and process URA private residential property data.

Includes functions to filter by property type, convert date columns,
compute property age and age bins, convert coordinate systems,
and prepare data for analysis by handling datatypes and dropping
unnecessary columns.
"""

from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

INPUT_CSV_PATH = Path("./src/data/input/raw_ura_data.csv")
OUTPUT_CSV_PATH = Path("./src/data/output/clean_ura_data.csv")


def filter_property_type(input_csv_path, property_type):
    """
    Filter the property DataFrame to include only rows matching the specified property type.

    :param input_csv_path: Path to the raw input CSV file containing property data
    :type input_csv_path: str or Path
    :param property_type: Property type to filter on (e.g., 'Condominium')
    :type property_type: str
    :return: Filtered DataFrame containing only the specified property type
    :rtype: pd.DataFrame
    """
    df_property = pd.read_csv(input_csv_path)

    df_condo = df_property[df_property["propertyType"] == property_type]

    return df_condo.reset_index(drop=True)


def to_datetime(df_property, col, fmt, new_col):
    """
    Convert a column in the DataFrame to datetime and sort by this new datetime column.

    :param df_property: DataFrame containing property data
    :type df_property: pd.DataFrame
    :param col: Name of the column to convert to datetime
    :type col: str
    :param fmt: Datetime format string to parse the column
    :type fmt: str
    :param new_col: Name of the new datetime column to create
    :type new_col: str
    :return: DataFrame sorted by the new datetime column
    :rtype: pd.DataFrame
    """
    df_property[new_col] = pd.to_datetime(df_property[col], format=fmt)

    df_property = df_property.sort_values(new_col, ascending=True)

    return df_property


def compute_property_age(df_property):
    """
    Compute property age bin from the tenure column. The bin for 'Freehold' remains
    as 'Freehold'.

    :param df_property: DataFrame containing property data with a 'tenure' column
    :type df_property: pd.DataFrame
    :return: DataFrame with added 'age_bin' column.
    :rtype: pd.DataFrame
    """

    df_property["tenure_start"] = df_property["tenure"].str.split().str[-1]

    def get_age(value):
        try:
            return datetime.now().year - int(value)
        except ValueError:
            return None

    df_property["age"] = df_property["tenure_start"].apply(get_age)
    df_with_age = df_property.dropna(subset=["age"]).copy()
    df_with_age["age_bin"] = pd.cut(df_with_age["age"], bins=10)

    df_freehold = df_property[df_property["age"].isnull()].copy()
    df_freehold["age_bin"] = "Freehold"

    return pd.concat([df_with_age, df_freehold], ignore_index=True)


def compute_property_tenure(df_property):
    """
    Compute property tenure bin from the tenure column. The bin for 'Freehold' remains
    as 'Freehold'.

    :param df_property: DataFrame containing property data with a 'tenure' column
    :type df_property: pd.DataFrame
    :return: DataFrame with added 'tenure_bin' column.
    :rtype: pd.DataFrame
    """

    df_property["tenure_bin"] = (
        df_property["tenure"].str.split().str[0:2].str.join(" ")
    )

    return df_property


def convert_svy21_to_wgs84(df_property, x_col, y_col):
    """
    Convert SVY21 coordinates (EPSG:3414) to WGS84 latitude and longitude (EPSG:4326).

    :param df_property: DataFrame containing property data with SVY21 x and y coordinates
    :type df_property: pd.DataFrame
    :param x_col: Name of the x coordinate column
    :type x_col: str
    :param y_col: Name of the y coordinate column
    :type y_col: str
    :return: DataFrame with added 'latitude' and 'longitude' columns in WGS84
    :rtype: pd.DataFrame
    """

    geometry = [
        Point(xy) for xy in zip(df_property[x_col], df_property[y_col])
    ]
    gdf = gpd.GeoDataFrame(
        df_property, geometry=geometry, crs="EPSG:3414"
    ).to_crs("EPSG:4326")

    df_with_lat_long = df_property.copy()
    df_with_lat_long["latitude"] = gdf.geometry.y
    df_with_lat_long["longitude"] = gdf.geometry.x

    return df_with_lat_long


def compute_days_since_1st_trans(df_property):
    """
    Calculate the number of days since the earliest transaction date.

    :param df_property: DataFrame containing a datetime column 'contract_date_dt'
    :type df_property: pd.DataFrame
    :return: DataFrame with new column 'days_since_1st_trans'
    :rtype: pd.DataFrame
    """

    first_date = df_property["contract_date_dt"].min()

    df_property["days_since_1st_trans"] = (
        df_property["contract_date_dt"] - first_date
    ).dt.days

    return df_property


def to_categorical(df_property):
    """
    Convert all object dtype columns and specific columns to categorical dtype.

    :param df_property: DataFrame containing property data
    :type df_property: pd.DataFrame
    :return: DataFrame with categorical columns
    :rtype: pd.DataFrame
    """

    obj_cols = df_property.select_dtypes(include=["object"]).columns
    df_property[obj_cols] = df_property[obj_cols].astype("category")

    for col in ["typeOfSale", "district"]:
        if col in df_property.columns:
            df_property[col] = df_property[col].astype("category")

    return df_property


def drop_columns(df_property, cols):
    """
    Drop specified columns from the DataFrame if they exist.

    :param df_property: DataFrame containing property data
    :type df_property: pd.DataFrame
    :param cols: List of column names to drop
    :type cols: list
    :return: DataFrame with specified columns dropped
    :rtype: pd.DataFrame
    """

    cols_to_drop = [col for col in cols if col in df_property.columns]

    if cols_to_drop:
        df_property = df_property.drop(columns=cols_to_drop)

    return df_property


def clean_ura_data(
    input_csv_path=INPUT_CSV_PATH,
    output_csv_path=OUTPUT_CSV_PATH,
    property_type="Condominium",
):
    """
    Run the full cleaning pipeline on URA property data for a specified property type.

    Steps include filtering, datetime conversion, calculating property age,
    coordinate conversion, type casting, dropping unnecessary columns, and saving to CSV.

    :param input_csv_path: Path to the input CSV file
    :type input_csv_path: str or Path
    :param output_csv_path: Path to save the cleaned output CSV file
    :type output_csv_path: str or Path
    :param property_type: Property type to filter (default is 'Condominium')
    :type property_type: str
    :return: Cleaned DataFrame ready for analysis
    :rtype: pd.DataFrame
    """

    df_processed = filter_property_type(input_csv_path, property_type)

    # Convert contractDate to datetime
    df_processed = to_datetime(
        df_processed,
        col="contractDate",
        fmt="%m%y",
        new_col="contract_date_dt",
    )

    # Compute days since first transaction
    df_processed = compute_days_since_1st_trans(df_processed)

    # Drop rows with missing x, y coordinates
    df_processed = df_processed.dropna(subset=["x", "y"])

    # Create property age tenure bin
    df_processed = compute_property_age(df_processed)
    df_processed = compute_property_tenure(df_processed)

    # Convert x and y to lat long
    df_processed = convert_svy21_to_wgs84(df_processed, x_col="x", y_col="y")

    # Rename price to target price
    df_processed = df_processed.rename(columns={"price": "target_price"})
    cols = ["target_price"] + [
        col for col in df_processed.columns if col != "target_price"
    ]
    df_processed = df_processed[cols]

    # Convert all object dtype, "typeOfSale", "district", to categorical
    df_processed = to_categorical(df_processed)

    # Drop columns
    drop_cols = [
        "propertyType",  # all values are "Condominium"
        "contractDate",  # replaced by 'contract_date_dt'
        "tenure",  # replaced by tenure_bin and age_bin
        "nettPrice",  # >99% missing
        "tenure_start",  # used to compute age
        "age",  # used to compute age_bin
        "x",  # replaced by "longitude"
        "y",  # replaced by "latitude"
    ]

    df_processed = drop_columns(df_processed, cols=drop_cols)

    # Prevent excel from interpreting "floorRange" as a date in csv format
    df_processed["floorRange"] = "Floor " + df_processed["floorRange"].astype(
        str
    )
    df_processed.to_csv(output_csv_path, index=False)

    return df_processed
