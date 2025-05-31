"""
Pipeline runner script.

This module executes the data cleaning function and prepares the dataset
for downstream unsupervised and supervised learning tasks.
"""

from src.clean_ura_data import clean_ura_data
from src.find_nearest_train_stn import find_nearest_train_stn
from src.clean_google_data import clean_google_data
from src.clean_prepare_population_data import (
    clean_population_data,
    prepare_population_data,
)
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
    df_population_clean = clean_population_data()
    df_monthly_population_growth_rates = prepare_population_data(df_population_clean)

    # TO-DO: Left join secondary data to df_ura_clean


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
