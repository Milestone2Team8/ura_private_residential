"""
Cleans and combines raw POI data from multiple CSV files into a GeoDataFrame.
"""
from datetime import datetime
from pathlib import Path
from shapely.geometry import Point
import pandas as pd
import geopandas as gpd

def extract_fetch_date(path):
    """
    Extracts the fetch date from a file path assuming the filename ends with a date 
    in the format 'ddmmyy', e.g., 'raw_google_data_school_010124.csv'.

    Args:
        path (Path): A pathlib.Path object representing the file path.

    Returns:
        datetime.datetime or None: The extracted date if parsing succeeds, 
        otherwise None.
    """
    try:
        return datetime.strptime(path.stem.split('_')[-1], '%d%m%y')
    except ValueError:
        return None

def clean_google_data(poi_type_list):
    """
    Loads, processes, and combines raw Google POI data for given POI types.
    
    For each POI type, the function:
    - Reads single or multiple dated CSV files.
    - Computes the change in user ratings over time if multiple dated files are present.
    - Adds metadata such as delta_rating_count, delta_time, and poi_type.
    - Converts the combined DataFrame into a GeoDataFrame with point geometries.
    - Saves the cleaned GeoDataFrame to a pickle file for reuse.

    Args:
        poi_type_list (list of str): List of POI types to process (e.g., 
        ['restaurant', 'school']).

    Returns:
        geopandas.GeoDataFrame: Combined and cleaned POI data with 
        geometry and calculated fields.

    Raises:
        ValueError: If no valid input files are found for any POI type.
    """
    combined_gdfs = []

    for poi_type in poi_type_list:
        input_files = sorted(Path("src/data/input").glob(
            f"raw_google_data_{poi_type}_*.csv"))
        base_file = Path(f"src/data/input/raw_google_data_{poi_type}.csv")

        if not input_files and not base_file.exists():
            print(f"Warning: No files found for POI type '{poi_type}'")
            continue

        if len(input_files) < 2:
            if input_files:
                df = pd.read_csv(input_files[0])
            else:
                df = pd.read_csv(base_file)

            df["delta_rating_count"] = 0
            df["delta_time"] = 0
            df["poi_type"] = poi_type
            combined_gdfs.append(df)
            continue

        files_with_dates = [(f, extract_fetch_date(f)) for f in input_files
                            if extract_fetch_date(f)]
        files_with_dates = sorted(files_with_dates, key=lambda x: x[1])

        df_min = pd.read_csv(files_with_dates[0][0]).set_index("place_id")
        df_max = pd.read_csv(files_with_dates[-1][0]).set_index("place_id")

        df_delta = df_max.copy()
        df_delta = df_delta.join(df_min[["user_ratings_total"]], rsuffix="_min",
                                 how="left")

        df_delta["delta_rating_count"] = (
            df_delta["user_ratings_total"] - df_delta["user_ratings_total_min"]
        ).fillna(0)

        df_delta["delta_time"] = (files_with_dates[-1][1] - files_with_dates[0][1]).days
        df_delta["poi_type"] = poi_type
        df_delta.reset_index(inplace=True)

        combined_gdfs.append(df_delta)

    if not combined_gdfs:
        raise ValueError("No valid input files found.")

    df_all = pd.concat(combined_gdfs, ignore_index=True)
    df_all = df_all.drop_duplicates(subset="place_id")

    geometry = [Point(xy) for xy in zip(df_all['lng'], df_all['lat'])]
    gdf = gpd.GeoDataFrame(df_all, geometry=geometry, crs="EPSG:4326")

    output_path = Path("src/data/output/clean_google_data_combined.pkl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_pickle(output_path)

    return gdf
