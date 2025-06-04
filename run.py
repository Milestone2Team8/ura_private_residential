"""
Pipeline runner script.

This module executes the data cleaning function and prepares the dataset
for downstream unsupervised and supervised learning tasks.
"""
import pandas as pd
from src.clean_ura_data import clean_ura_data
from src.find_nearest_train_stn import find_nearest_train_stn
from src.clean_google_data import clean_google_data
from src.find_nearest_google_data import find_nearby_google_poi
from src.clean_prepare_population_data import (
    clean_population_data,
    prepare_population_data
)
from src.clean_prepare_marriage_data import (
    clean_marriage_data,
    prepare_marriage_data
)
from src.clean_prepare_cpi_data import (
    clean_cpi_data,
    prepare_cpi_data
)
from src.clean_prepare_sora_data import (
    clean_sora_data,
    prepare_sora_data
)
from src.clean_prepare_property_index_data import (
    clean_property_index_data,
    prepare_property_index_data
)
from src.utils.secondary_ds_helper_functions import concat_and_filter_by_date

# pylint: disable=unused-variable


def run_all(poi_type_list):
    """
    Cleans raw URA private residential data and Google POI data
    and prepares them for modeling tasks.

    Args:
        poi_type_list (list of str): List of POI types to clean and combine.

    Returns:
        tuple: Cleaned URA dataframe and combined Google POI GeoDataFrame.
    """
    df_clean = clean_ura_data()
    df_nearest_mrt, df_nearest_lrt = find_nearest_train_stn()
    df_google_clean = clean_google_data(poi_type_list)
    df_google_nearest = find_nearby_google_poi(poi_type_list)

    df_ecosocial = run_secondary()

    df_clean['contract_month'] = pd.to_datetime(df_clean['contract_date_dt']).dt.to_period('M')
    df_ecosocial['ecosocial_month'] = pd.to_datetime(df_ecosocial.index).to_period('M')
    df_merged = pd.merge(
        df_clean,
        df_ecosocial.reset_index().assign(ecosocial_month=pd. \
            to_datetime(df_ecosocial.index).to_period('M')),
        left_on='contract_month',
        right_on='ecosocial_month',
        how='left'
    )

    df_merged.drop(columns=['contract_month', 'ecosocial_month'], inplace=True)
    df_merged.to_csv("./src/data/output/clean_merged_ura_data.csv", index=False)

def run_secondary():
    """
    Runs secondary dataset pipeline jobs
    """
    df_monthly_population_growth_rates = prepare_population_data(
        clean_population_data(),
        "2020",
        "2025"
    )

    df_monthly_marriage_growth_rates = prepare_marriage_data(
        clean_marriage_data(),
        "2020",
        "2025"
    )

    df_monthly_cpi = prepare_cpi_data(clean_cpi_data(),
        "2019-12-01",
        "2025-12-01"
    )

    df_monthly_sora = prepare_sora_data(
         clean_sora_data(),
         "2019-12-31",
         "2025-12-01"
    )

    df_monthly_property_index = prepare_property_index_data(
        clean_property_index_data(),
        "2019-12-01",
        "2025-04-01"
    )

    return concat_and_filter_by_date([df_monthly_population_growth_rates,
            df_monthly_marriage_growth_rates,
            df_monthly_cpi,
            df_monthly_sora,
            df_monthly_property_index], "month", "2020-01-01", "2025-05-30")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run data cleaning pipeline")
    parser.add_argument(
        "--poi_type_list",
        nargs="+",
        required=True,
        help="List of POI types to include (e.g., restaurant school pharmacy)",
    )
    args = parser.parse_args()

    run_all(poi_type_list=args.poi_type_list)
