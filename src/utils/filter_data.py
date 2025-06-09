"""
Filter DataFrame rows where the date_column is between start_date and end_date inclusive
"""

import pandas as pd


def filter_between_dates(df_ura, date_column, start_date, end_date):
    """
    Filter DataFrame rows where the date_column is between start_date and end_date inclusive.

    :param df_ura: Input DataFrame
    :type df_ura: pd.DataFrame
    :param date_column: Name of the column containing dates
    :type date_column: str
    :param start_date: Start date (inclusive), can be string or pd.Timestamp
    :param end_date: End date (inclusive), can be string or pd.Timestamp
    :return: Filtered DataFrame with rows between start_date and end_date
    :rtype: pd.DataFrame
    """
    mask = (df_ura[date_column] >= pd.to_datetime(start_date)) & (
        df_ura[date_column] <= pd.to_datetime(end_date)
    )
    return df_ura.loc[mask].copy()
