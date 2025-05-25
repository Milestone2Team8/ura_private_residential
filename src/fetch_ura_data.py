"""
Module to fetch and process private residential property transaction data
from the URA Data Service API.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd
import requests

OUTPUT_JSON_PATH = Path("./data/input/raw_ura_data.json")
OUTPUT_CSV_PATH = Path("./data/input/raw_ura_data.csv")


def fetch_private_residential_data(
    access_key,
    output_json_path=OUTPUT_JSON_PATH,
    output_csv_path=OUTPUT_CSV_PATH,
):
    """
    Fetches private residential property transaction data from URA Data Service API.

    :param access_key: URA API access key for authentication
    :type access_key: str
    :param output_json_path: Path to save combined raw JSON data
    :type output_json_path: pathlib.Path or str
    :param output_csv_path: Path to save normalized CSV data
    :type output_csv_path: pathlib.Path or str
    :return: DataFrame containing private residential transaction data
    :rtype: pd.DataFrame
    """

    # Get token using access key
    user_agent = "PostmanRuntime/7.28.4"
    headers = {"AccessKey": access_key, "User-Agent": user_agent}
    token_response = requests.get(
        "https://eservice.ura.gov.sg/uraDataService/insertNewToken/v1",
        headers=headers,
        timeout=60,
    )
    token = token_response.json()["Result"]

    # Get data with access key and token
    headers["Token"] = token
    batch_urls = [
        (
            "https://eservice.ura.gov.sg/uraDataService/invokeUraDS/v1?"
            "service=PMI_Resi_Transaction&batch=1"
        ),
        (
            "https://eservice.ura.gov.sg/uraDataService/invokeUraDS/v1?"
            "service=PMI_Resi_Transaction&batch=2"
        ),
        (
            "https://eservice.ura.gov.sg/uraDataService/invokeUraDS/v1?"
            "service=PMI_Resi_Transaction&batch=3"
        ),
        (
            "https://eservice.ura.gov.sg/uraDataService/invokeUraDS/v1?"
            "service=PMI_Resi_Transaction&batch=4"
        ),
    ]

    all_results = []
    for _, url in enumerate(batch_urls, start=1):
        response = requests.get(url, headers=headers, timeout=60)
        result = response.json().get("Result", [])
        all_results.extend(result)

    # Save combined JSON to file
    all_jsons = {"results": all_results}
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(all_jsons, f, indent=4)

    # Normalize all transactions
    df_all = pd.json_normalize(
        all_results,
        record_path="transaction",
        meta=["street", "project", "marketSegment", "x", "y"],
        errors="ignore",
    )

    # Save DataFrame to file
    df_all.to_csv(output_csv_path, index=False)

    return df_all


if __name__ == "__main__":
    env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(dotenv_path=env_path)
    fetch_private_residential_data(access_key=os.getenv("URA_ACCESS_KEY"))
