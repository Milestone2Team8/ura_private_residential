"""
Module to fetch the latitude and longitude of MRT and LRT trains in Singapore
from the OneMap API service.
"""

import json
import logging
from pathlib import Path
import urllib.parse

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INPUT_XLS_PATH = Path("./src/data/input/raw_train_stn.xls")
OUTPUT_CSV_PATH = Path("./src/data/output/onemap_train_stn_lat_long.csv")


def get_unique_lst(input_xls_path, columns):
    """
    Returns a list of unique string values formed by concatenating the specified
    columns. If only one column is provided, returns
    its unique values as strings.

    :param df_property: DataFrame containing amenity data.
    :type df_property: pd.DataFrame
    :param columns: List of column names to concatenate.
    :type columns: list[str]
    :return: Unique strings from the specified columns.
    :rtype: list[str]
    """
    df_amenity = pd.read_excel(input_xls_path)

    concatenated_column = df_amenity[columns].astype(str).agg(" ".join, axis=1)
    unique_lst = concatenated_column.unique().tolist()

    return unique_lst


def fetch_lat_long(search_vals, output_csv=OUTPUT_CSV_PATH):
    """
    Searches and returns the longitudes and latitudes via the OneMAP API.

    :param search_vals: List of values to search
    :type search_vals: list[str]
    :return: DataFrame containing the longitudes and latitudes via the OneMAP API.
    :rtype: pd.DataFrame
    """
    for index, search_val in enumerate(search_vals):
        # Encode special characters in the URL properly (e.g. "+Bugis")
        encoded_search_val = urllib.parse.quote(str(search_val))

        url = (
            f"https://www.onemap.gov.sg/api/common/elastic/search?"
            f"searchVal={encoded_search_val}&returnGeom=Y&getAddrDetails=Y"
        )

        response = requests.get(url, timeout=60)
        try:
            dict_response = json.loads(response.text)
        except ValueError:
            print("JSONDecodeError")

        if dict_response["results"]:
            df_result = pd.DataFrame.from_dict(dict_response["results"])
        else:
            df_result = pd.DataFrame(
                {
                    "SEARCHVAL": ["NOT_FOUND"],
                    "BLK_NO": ["NOT_FOUND"],
                    "ROAD_NAME": ["NOT_FOUND"],
                    "BUILDING": ["NOT_FOUND"],
                    "ADDRESS": ["NOT_FOUND"],
                    "POSTAL": ["NOT_FOUND"],
                    "X": ["NOT_FOUND"],
                    "Y": ["NOT_FOUND"],
                    "LATITUDE": ["NOT_FOUND"],
                    "LONGITUDE": ["NOT_FOUND"],
                }
            )

        df_result["org_search_val"] = search_val

        if index == 0:
            df_results = df_result
        else:
            df_results = pd.concat([df_results, df_result], ignore_index=False)

    df_results.columns = df_results.columns.str.lower()

    dups_count = df_results.duplicated(subset=["org_search_val"]).sum()
    if dups_count > 0:
        logger.warning(
            "OneMap API returned multiple results for some search values."
        )

    df_results.to_csv(output_csv, index=False)

    return df_results


if __name__ == "__main__":
    unique_train_stn = get_unique_lst(
        INPUT_XLS_PATH, ["mrt_station_english", "stn_code"]
    )
    fetch_lat_long(unique_train_stn)
