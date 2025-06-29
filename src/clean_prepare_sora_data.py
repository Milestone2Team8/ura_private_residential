"""
Module to clean and process sora rates dataset.

Process includes functions to:
- Convert daily rates to monthly rates.
"""

from pathlib import Path
import pandas as pd

INPUT_SORA_PATH = Path("./src/data/input/Domestic_Interest_Rates.csv")

def clean_sora_data(input_path=INPUT_SORA_PATH):
    """
    Apply data cleaning to SORA dataframes

    :param input_path: Path to the input population xlsx file
    :type input_path: str or Path
    :return: Cleaned DataFrame ready for analysis
    :rtype: pd.DataFrame
    """

    df_sora = pd.read_csv(input_path)
    df_sora_clean = df_sora.copy()
    try:
        df_sora_clean.columns = (
            ['Value Date', 'Unnamed 1', 'Index', 'Publication Date', 'SORA']
        )
        df_sora_clean = df_sora_clean.drop(columns=['Unnamed 1', 'Index'])
        df_sora_clean = df_sora_clean.drop(0).reset_index(drop=True)
        df_sora_clean['Publication Date'] = df_sora_clean['Publication Date'].\
            str.replace('Sept', 'Sep', regex=False)

        df_sora_clean['Publication Date'] = (
            pd.to_datetime(
                df_sora_clean['Publication Date'],
                format='%d-%b-%y',
                errors='coerce')
        )
        df_sora_clean['SORA'] = pd.to_numeric(df_sora_clean['SORA'], errors='coerce')
        df_sora_clean.set_index('Publication Date', inplace=True)
    except Exception as e:
        raise e

    return df_sora_clean


def prepare_sora_data(df_clean : pd.DataFrame, start_date : str = "2019-01-01" ,
    end_date : str = "2025-05-12"):
    """
    Filters a DataFrame by date index, resamples to month-end frequency, 
    and formats the index.

    :param df_clean: DataFrame with a datetime index
    :type df_clean: pd.DataFrame
    :param start_date: Start date (inclusive) in 'YYYY-MM-DD' format, or None to skip filtering
    :type start_date: str or None
    :param end_date: End date (inclusive) in 'YYYY-MM-DD' format, or None to skip filtering
    :type end_date: str or None
    :return: Manipulated and resampled dataframe ready for analysis
    :rtype: pd.DataFrame
    """
    if start_date:
        df_clean = df_clean[df_clean.index >= start_date]
    if end_date:
        df_clean = df_clean[df_clean.index <= end_date]
    df_monthly_sora = df_clean.resample('ME').last()

    df_monthly_sora.drop(columns=['Value Date'], inplace=True)
    df_monthly_sora.index = df_monthly_sora.index.to_period('M')
    df_monthly_sora = df_monthly_sora.reset_index()
    df_monthly_sora = df_monthly_sora.rename(columns={'Publication Date': 'month'})
    return df_monthly_sora
