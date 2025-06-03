"""
Module to clean and process marriage dataset.

Process includes functions to:
- Predict missing year 2023/2024
- Convert yearly rates to monthly rates.
"""

from pathlib import Path
import pandas as pd
from src.utils.secondary_ds_helper_functions import (clean_singstat_ds,
                        clean_and_prepare_dataset, predict_missing_data,
                        distribute_yearly_to_monthly_rate)

INPUT_MARRIAGE_PATH = Path("./src/data/input/M830102.xlsx")

def clean_marriage_data(input_path=INPUT_MARRIAGE_PATH):
    """
    Apply data cleaning to marriage growth Singstat dataframes

    :param input_path: Path to the input population xlsx file
    :type input_path: str or Path
    :return: Cleaned DataFrame ready for analysis
    :rtype: pd.DataFrame
    """

    df_marriage = pd.read_excel(input_path, skiprows=8)
    df_marriage_clean = clean_singstat_ds(df_marriage)

    return df_marriage_clean


def prepare_marriage_data(df_clean : pd.DataFrame, start_year: str = "2020",
                        end_year:str = "2025"):

    """
    Cleans yearly marriage growth dataframe, and convert
    yearly rate to monthly.
    
    :param df_clean: Dataframe to be prepared for analysis
    :type df_clean: pd.DataFrame
    :param start_year: Start year (inclusive) in YYYY
    :type start_date: str
    :param end_year: End year (inclusive) in YYYY
    :type end_year: str
    :return: Manipulated and resampled dataframe ready for analysis
    :rtype: pd.DataFrame
    """

    df_yearly_marriage_rates = clean_and_prepare_dataset(
        df_clean,
        "Crude Marriage Rate (Per 1,000 Residents)",
        "marriage_crude_rate",
    )

    df_yearly_marriage_rates = predict_missing_data(
        df_yearly_marriage_rates, "year_index", "marriage_crude_rate",
        int(end_year) + 1
    )

    df_monthly_marriage_rates = distribute_yearly_to_monthly_rate(
        df_yearly_marriage_rates, "marriage_crude_rate", int(start_year),
        int(end_year)
    )

    df_monthly_marriage_rates = df_monthly_marriage_rates.reset_index()

    return df_monthly_marriage_rates
