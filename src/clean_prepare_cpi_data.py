"""
Module to clean and process cpi dataset.

Process includes functions to:
- Calculate adjusted cpi rate using Housing & Utilities key
- 
"""

from pathlib import Path
import pandas as pd
from src.secondary_ds_helper_functions import (clean_singstat_ds,
                       prepare_housing_cpi)

INPUT_CPI_PATH = Path("./src/data/input/M213751.xlsx")

def clean_cpi_data(input_csv_path=INPUT_CPI_PATH):
    """
    Apply data cleaning to cpi rate Singstat dataframes

    :param input_csv_path: Path to the input population xlsx file
    :type input_csv_path: str or Path
    :return: Cleaned DataFrame ready for analysis
    :rtype: pd.DataFrame
    """

    df_cpi = pd.read_excel(input_csv_path, skiprows=9)
    df_cpi_clean = clean_singstat_ds(df_cpi)

    return df_cpi_clean


def prepare_cpi_data(df_clean : pd.DataFrame):

    """
    Cleans yearly cpi rates dataframe
    
    :param df_clean: Dataframe to be prepared for analysis
    :type df_clean: pd.DataFrame
    :return: Manipulated dataframe ready for analysis
    :rtype: pd.DataFrame
    """

    df_monthly_cpi = prepare_housing_cpi(df_clean)

    return df_monthly_cpi
