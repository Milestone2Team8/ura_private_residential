"""
This module merges 2 datasets (ura, amenities) using latitude and longitude.
"""

import pandas as pd


def merge_amenities_data(df_ura, df_amenities, poi_type_list):
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

<<<<<<< HEAD
    amenities_cols = ["mrt_nearest_distance_m", "lrt_nearest_distance_m"] + [
=======
    amenities_cols = [
        "mrt_nearest_distance_m",
        "lrt_nearest_distance_m"
    ] + [
>>>>>>> 23e0a66 (fix merge_amenities_data())
        col
        for poi_type in poi_type_list
        for col in [
            f"poi_count_{poi_type}",
            f"sum_user_ratings_{poi_type}",
            f"avg_rating_{poi_type}",
            f"avg_price_level_{poi_type}",
<<<<<<< HEAD
            f"price_level_obs_count_{poi_type}",
=======
            f"price_level_obs_count_{poi_type}"
>>>>>>> 23e0a66 (fix merge_amenities_data())
        ]
    ]

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