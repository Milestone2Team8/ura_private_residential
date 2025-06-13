"""
Convert columns in cat_features from pd.Interval or categorical dtype to string.
"""

import pandas as pd


def convert_interval_to_str(df_input, cat_features):
    """
    Convert columns in cat_features from pd.Interval or categorical dtype to string.

    :param df: pandas DataFrame
    :param cat_features: list of categorical feature column names
    :return: DataFrame with converted categorical columns
    """
    df_input = df_input.copy()
    for col in cat_features:
        if col in df_input.columns:
            if pd.api.types.is_categorical_dtype(df_input[col]) or (
                not df_input[col].dropna().empty
                and isinstance(df_input[col].dropna().iloc[0], pd.Interval)
            ):
                df_input[col] = df_input[col].astype(str)
    return df_input
