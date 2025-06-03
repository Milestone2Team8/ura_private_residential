"""
Module to clean and process population dataset.

Process includes functions to:
- Select population data only
- Compute monthly population growth rates by percentage
- Predict missing years data
- Do temporal data processing and transform into monthly rates
"""

from pathlib import Path
import pandas as pd
from src.secondary_ds_helper_functions import (clean_singstat_ds,
                        clean_and_prepare_dataset, predict_missing_data,
                        distribute_yearly_to_monthly_rate)


INPUT_POPULATION_PATH = Path("./src/data/input/M810001.xlsx")

def clean_population_data(input_path=INPUT_POPULATION_PATH):
    """
    Apply data cleaning to population growth Singstat dataframes

    :param input_path: Path to the input population xlsx file
    :type input_path: str or Path
    :return: Cleaned DataFrame ready for analysis
    :rtype: pd.DataFrame
    """

    df_population = pd.read_excel(input_path, skiprows=8)
    df_population_clean = clean_singstat_ds(df_population)

    return df_population_clean


def prepare_population_data(df_clean : pd.DataFrame):

    """
    Cleans yearly population growth dataframe, and convert
    yearly rate to monthly.
    
    :param df_clean: Dataframe to be prepared for analysis
    :type df_clean: pd.DataFrame
    :return: Manipulated and resampled dataframe ready for analysis
    :rtype: pd.DataFrame
    """
    df_yearly_population_growth_rates = clean_and_prepare_dataset(
        df_clean, "Total Population (Number)", "population"
    )

    df_yearly_population_growth_rates["population"] = pd.to_numeric(
        df_yearly_population_growth_rates["population"], errors="coerce"
    )

    df_yearly_population_growth_rates.dropna(inplace=True)

    df_yearly_population_growth_rates = predict_missing_data(
        df_yearly_population_growth_rates, "year_index", "population", 2026
    )

    df_yearly_population_growth_rates["population_growth_rate"] = (
        df_yearly_population_growth_rates["population"].pct_change()
    )

    df_monthly_population_growth_rates = distribute_yearly_to_monthly_rate(
        df_yearly_population_growth_rates, "population_growth_rate", 2020, 2025
    )

    return df_monthly_population_growth_rates
