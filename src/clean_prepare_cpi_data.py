"""
Module to clean and process cpi dataset.

Process includes functions to:
- Calculate adjusted cpi rate using Housing & Utilities key
- 
"""

from pathlib import Path
import pandas as pd
from src.utils.secondary_ds_helper_functions import (clean_singstat_ds,
                       clean_and_prepare_dataset,
                       predict_missing_data)

INPUT_CPI_PATH = Path("./src/data/input/M213751.xlsx")

def clean_cpi_data(input_path=INPUT_CPI_PATH):
    """
    Apply data cleaning to cpi rate Singstat dataframes

    :param input_path: Path to the input population xlsx file
    :type input_path: str or Path
    :return: Cleaned DataFrame ready for analysis
    :rtype: pd.DataFrame
    """

    df_cpi = pd.read_excel(input_path, skiprows=9)
    df_cpi_clean = clean_singstat_ds(df_cpi)

    return df_cpi_clean


def prepare_cpi_data(df_clean : pd.DataFrame, start_date: str = "2019-12-01",
                        end_date:str = "2025-06-01"):

    """
    Process yearly cpi rates dataframe
    
    :param df_clean: Dataframe to be prepared for analysis
    :type df_clean: pd.DataFrame
    :param start_date: Start date (inclusive) in YYYY-MM-DD
    :type start_date: str
    :param end_date: End date (inclusive) in YYYY-MM-DD
    :type end_date: str
    :return: Manipulated and resampled dataframe ready for analysis
    :rtype: pd.DataFrame
    """

    df_clean_housing = clean_and_prepare_dataset(
        df_clean, "  Housing & Utilities", "cpi_housing", "month",
            options={"is_monthly": True}
    )

    try:

        start_date_dt = pd.to_datetime(start_date)
        base_date_dt = start_date_dt - pd.offsets.MonthEnd(1)

        df_clean_housing["month"] = pd.to_datetime(
            df_clean_housing["month"].str.strip(), format="%Y %b", errors="coerce"
        )

        df_monthly_cpi = df_clean_housing[
            (df_clean_housing["month"] >= start_date) &
            (df_clean_housing["month"] <= end_date)
        ].reset_index(drop=True)

        df_monthly_cpi["month"] = df_monthly_cpi["month"] - pd.offsets.MonthEnd(1)

        base_cpi = (
            df_monthly_cpi[df_monthly_cpi["month"] == base_date_dt]["cpi_housing"].values[0]
        )
        df_monthly_cpi["cpi_adjusted"] = df_monthly_cpi["cpi_housing"] / base_cpi
        df_monthly_cpi.drop(columns=["Data Series"], inplace=True)
        df_monthly_cpi.set_index("month", inplace=True)
        df_monthly_cpi.sort_index(inplace=True)

        df_monthly_cpi = df_monthly_cpi.apply(pd.to_numeric)

        df_monthly_cpi["cpi"] = df_monthly_cpi["cpi_adjusted"].pct_change()
        df_monthly_cpi.drop(columns=["cpi_housing"], inplace=True)
        df_monthly_cpi.drop(columns=["cpi_adjusted"], inplace=True)
        start_date_pd = pd.to_datetime(start_date)
        end_of_next_month = (start_date_pd + pd.offsets.MonthEnd(2)).strftime('%Y-%m-%d')
        df_monthly_cpi = df_monthly_cpi[df_monthly_cpi.index >= end_of_next_month]
        df_monthly_cpi.index = df_monthly_cpi.index.to_period("M")
        df_monthly_cpi = predict_missing_data(
            df_monthly_cpi, None , "cpi",
            int(str(pd.to_datetime(end_date).year)) + 1
        )

        return df_monthly_cpi
    except Exception as e:
        raise e
