"""
Split data into train and test sets by date, preserving time order.
"""


def split_time_series_train_test(
    df_ura, date_column="contract_date_dt", test_size=0.2
):
    """Split data into train and test sets by date, preserving time order."""
    df_sorted = df_ura.sort_values(date_column).copy()
    split_idx = int(len(df_sorted) * (1 - test_size))
    df_train = df_sorted.iloc[:split_idx]
    df_test = df_sorted.iloc[split_idx:]

    return df_train, df_test
