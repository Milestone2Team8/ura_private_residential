"""
Module to clean and process marriage dataset.

Process includes functions to:
- Convert quarterly rates to monthly rates.
"""

from pathlib import Path
import pandas as pd
from src.utils.secondary_ds_helper_functions import (
                    parse_quarter, distribute_quarterly_to_monthly_rate,
                    predict_missing_data
                    )


INPUT_PROPERTY_INDEX_PATH = Path("./src/data/input/Private Residential Property Price Index.csv")

def clean_property_index_data(input_path=INPUT_PROPERTY_INDEX_PATH) -> pd.DataFrame:
    """
    Cleans the private property index dataset by:
    
    :param input_path: Path to the property index CSV
    :type input_path: str or Path
    :return: Cleaned DataFrame
    :rtype: pd.DataFrame
    """
    try:
        df_property_index = pd.read_csv(input_path)

        df_property_index = df_property_index[df_property_index["property_type"] == "Non-Landed"]

        df_property_index.rename(columns={"index": "price_index"}, inplace=True)
        df_property_index["quarter"] = df_property_index["quarter"].str.strip()
        df_property_index["month"] = df_property_index["quarter"].\
            apply(lambda x: parse_quarter(x, separator="-"))
        df_property_index.set_index("month", inplace=True)
        df_property_index.sort_index(inplace=True)
        return df_property_index
    except Exception as e:
        raise e


def prepare_property_index_data( df_clean: pd.DataFrame, start_date: str = "2019-12-01",
            end_date: str = "2025-12-01") -> pd.DataFrame:
    """
    Filters private property index data by date range and calculates
    delta (change) from the previous period.

    :param df_clean: Cleaned non-landed property index DataFrame
    :type df_clean: pd.DataFrame
    :param start_date: Start date (inclusive) in YYYY-MM-DD
    :type start_date: str
    :param end_date: End date (inclusive) in YYYY-MM-DD
    :type end_date: str
    :return: DataFrame with monthly deltas
    :rtype: pd.DataFrame
    """
    try:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        df_quarterly_index = df_clean[(df_clean.index >= start_dt) \
            & (df_clean.index <= end_dt)].copy()
        df_quarterly_index["rate"] = df_quarterly_index["price_index"].pct_change()
        df_quarterly_index.dropna(inplace=True)
        df_quarterly_index.index = df_quarterly_index.index.to_period("M")
        df_monthly_index = distribute_quarterly_to_monthly_rate(
            df_quarterly_index,
            "rate", 
            "price_index",
            "2019-12-01", 
            "2025-04-01"
        )
        df_monthly_index = df_monthly_index.drop(columns=['quarterly_rate'])
        df_monthly_index = df_monthly_index.drop(columns=['monthly_rate'])

        df_monthly_index = predict_missing_data(
            df_monthly_index, None , "monthly_price_index",
            int(str(pd.to_datetime(end_date).year)) + 1
        )
        df_monthly_index.reset_index()
        return df_monthly_index
    except Exception as e:
        raise e
