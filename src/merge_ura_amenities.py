"""
This module merges 2 datasets (ura, amenities) using latitude and longitude.
"""

import pandas as pd

from src.utils.load_configs import load_configs


def merge_amenities_data(df_ura, df_amenities):
    """
    Merges ura and amenities datasets.

    :param df_ura: primary dataframe
    :type df_ura: pd.Dataframe
    :param df_amenities: secondary dataframe
    :type df_amenities: pd.Dataframe
    """
    rows_before = len(df_ura)
    df_merged = df_ura.copy()

    df_merged["longitude"] = df_merged["longitude"].round(8)
    df_merged["latitude"] = df_merged["latitude"].round(8)

    configs = load_configs("features.yml")
    amenities_cols = configs["all_features"]["num_amenities"]

    for df_amenity in df_amenities:
        cols_to_keep = ["longitude", "latitude"] + [
            col for col in amenities_cols if col in df_amenity.columns
        ]
        df_amenity = df_amenity[cols_to_keep].copy()
        df_amenity["longitude"] = df_amenity["longitude"].round(8)
        df_amenity["latitude"] = df_amenity["latitude"].round(8)
        df_merged = pd.merge(
            df_merged, df_amenity, how="left", on=["longitude", "latitude"]
        )

    rows_after = len(df_merged)

    assert rows_before == rows_after, (
        f"Row counts differ after left join with amenities data. "
        f"Rows Before: {rows_before}, Rows After: {rows_after}"
    )

    return df_merged
