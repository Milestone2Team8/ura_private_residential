"""
Pipeline runner script.

This module executes the data cleaning function and prepares the dataset
for downstream unsupervised and supervised learning tasks.
"""

import logging
from pathlib import Path

from src.analysis.tsne_visualize import generate_plot_tsne_clusters
from src.analysis.unsupervised_kmeans import perform_kmeans
from src.analysis.ablation_analysis import perform_ablation_analysis
from src.clean_google_data import clean_google_data
from src.clean_prepare_cpi_data import clean_cpi_data, prepare_cpi_data
from src.clean_prepare_marriage_data import (
    clean_marriage_data,
    prepare_marriage_data,
)
from src.clean_prepare_population_data import (
    clean_population_data,
    prepare_population_data,
)
from src.clean_prepare_property_index_data import (
    clean_property_index_data,
    prepare_property_index_data,
)
from src.clean_prepare_sora_data import clean_sora_data, prepare_sora_data
from src.clean_ura_data import clean_ura_data
from src.detect_outliers import detect_outliers_generate_plots
from src.find_nearest_google_data import find_nearby_google_poi
from src.find_nearest_train_stn import find_nearest_train_stn
from src.merge_ura_amenities import merge_amenities_data
from src.merge_ura_ecosocial import merge_ecosocial_data
from src.normalize_sale_price import normalize_prices
from src.perform_failure_analysis import perform_failure_analysis
from src.run_time_series_cv import run_time_series_cv
from src.utils.secondary_ds_helper_functions import concat_and_filter_by_date
from src.utils.spilt_time_series_train_test import split_time_series_train_test
from src.utils.validate import validate_merge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# pylint: disable=unused-variable

OUTPUT_PATHS = {
    "clean_merged_data": Path("./src/data/output/clean_merged_ura_data.csv"),
    "all_models_results": Path(
        "./src/data/output/all_models_features_results.json"
    ),
    "primary_features_results": Path(
        "./src/data/output/best_model_primary_features_results.json"
    ),
    "amenities_features_results": Path(
        "./src/data/output/best_model_amenities_features_results.json"
    ),
    "ecosocial_features_results": Path(
        "./src/data/output/best_model_ecosocial_features_results.json"
    ),
    "sensitivity_results": Path(
        "./src/data/output/best_model_sensitivity_results.json"
    ),
}


def prepare_ecosocial_data():
    """Prepares economic social indicators datasets."""
    df_monthly_population_growth_rates = prepare_population_data(
        clean_population_data(), "2020", "2025"
    )

    df_monthly_marriage_growth_rates = prepare_marriage_data(
        clean_marriage_data(), "2020", "2025"
    )

    df_monthly_cpi = prepare_cpi_data(
        clean_cpi_data(), "2019-12-01", "2025-12-01"
    )

    df_monthly_sora = prepare_sora_data(
        clean_sora_data(), "2019-12-31", "2025-12-01"
    )

    df_monthly_property_index = prepare_property_index_data(
        clean_property_index_data(), "2019-12-01", "2025-04-01"
    )

    return concat_and_filter_by_date(
        [
            df_monthly_population_growth_rates,
            df_monthly_marriage_growth_rates,
            df_monthly_cpi,
            df_monthly_sora,
            df_monthly_property_index,
        ],
        "month",
        "2020-01-01",
        "2025-05-30",
    )


def prepare_amenities_data(poi_type_list):
    """Prepares amenities datasets."""
    df_nearest_mrt, df_nearest_lrt = find_nearest_train_stn()
    df_google_clean = clean_google_data(poi_type_list)
    df_google_nearest = find_nearby_google_poi(poi_type_list)

    return df_nearest_mrt, df_nearest_lrt, df_google_nearest


def prepare_merge_all_data(poi_type_list):
    """Prepares and merges primary and secondary datasets."""
    logger.info("---Cleaning and Processing Primary Dataset\n")
    df_primary_data = clean_ura_data()

    logger.info("---Merging Primary and Secondary Datasets\n")

    df_mrt, df_lrt, df_google = prepare_amenities_data(poi_type_list)
    df_merged_data = merge_amenities_data(
        df_primary_data, [df_mrt, df_lrt, df_google], poi_type_list
    )

    df_econ_data = prepare_ecosocial_data()
    df_merged_data = merge_ecosocial_data(df_merged_data, df_econ_data)

    df_merged_data = normalize_prices(df_merged_data)

    df_merged_data.to_csv(OUTPUT_PATHS["clean_merged_data"], index=False)
    validate_merge(df_primary_data, df_merged_data, df_name="Merged Dataset")

    return df_merged_data


def run_all(poi_type_list):
    """
    Cleans raw URA private residential data and Google POI data
    and prepares them for modeling tasks.

    :param poi_type_list: List of POI types to clean and combine.
    :type poi_type_list: list of str

    :return: Cleaned URA dataframe and combined Google POI GeoDataFrame.
    :rtype: tuple
    """

    # Prepare and merge primary with secondary data
    df_merged_data = prepare_merge_all_data(poi_type_list)

    # Unsupervised learning analysis
    df_kmeans, x_scaled, no_of_cluster = perform_kmeans(df_merged_data)
    generate_plot_tsne_clusters(df_kmeans, x_scaled, no_of_cluster)
    detect_outliers_generate_plots(df_merged_data)

    # Supervised learning analysis
    df_single_trans = df_merged_data[df_merged_data["noOfUnits"] == 1]
    df_train, df_test = split_time_series_train_test(df_single_trans)

    # --- All models and features ---
    best_pipeline, best_result = run_time_series_cv(
        df_train,
        mode="find_best_model",
        feature_set="all_features",
        output_path=OUTPUT_PATHS["all_models_results"],
    )

    # --- Best model feature ablation analysis ---
    feature_ablation_results = []

    for feature_set in [
        "primary_features",
        "amenities_features",
        "ecosocial_features",
    ]:
        _, feature_ablation_result = run_time_series_cv(
            df_train,
            mode="best_model_single_param",  # best model setting
            feature_set=f"{feature_set}",
            output_path=OUTPUT_PATHS[f"{feature_set}_results"],
        )

        feature_ablation_results.append(feature_ablation_result)

    # --- Best model sensitivity analysis ---
    _, sensitivity_result = run_time_series_cv(
        df_train,
        mode="best_model_multi_params",
        feature_set="all_features",
        output_path=OUTPUT_PATHS["sensitivity_results"],
    )

    # --- Best model failure analysis ---
    best_pipeline, _ = run_time_series_cv(
        df_train,
        mode="best_model_single_param",
        feature_set="all_features",
    )

    #perform_failure_analysis(best_pipeline, df_train, df_test, indices=[0, 1])

    # Ablation analysis
    perform_ablation_analysis(best_pipeline, df_train, df_test,
       target_column="target_price")

    # TO-DO
    # Please see the “Tips for Project Report” video under “Week 6 Project Check-in”

    # (6 points) Do a feature importance and ablation analysis on your best model to
    # get insight into which features are or are not contributing to prediction success/failure.
    # Jerome: The best_model_pipeline can be used to perform .fit on df_train and .predict on
    # test. to experiment with the different feature sets (e.g. primary data, amenities data
    # and econ data).

    # (4 points) Do at least one sensitivity analysis on your best model: How sensitive are your
    # results to choice of (hyper-)parameters, features, or other varying solution elements?
    # Jerome: Using the best model settings saved under .\data\output\csv_results.json, choose one
    # hyperparameter to analysis and plot the validation curve. See video under Supervised Learning
    # Week 2 - Cross Validation 6:58 min on how to plot the curve. Note: The best_model_pipeline
    # cannot be reused directly because hyperparameter experimentation is still needed.

    # Failure analysis (5 points)
    # Select at least 3 *specific* examples (records) where prediction failed, and analyze
    # possible reasons why.
    # Ideally you should be able to identify at least three different categories of failure.
    # What future improvements might fix the failures?
    # Jerome: Predict df_test using the best_model_pipeline and conduct analysis.


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run data cleaning pipeline")
    parser.add_argument(
        "--poi_type_list",
        nargs="+",
        default=[
            "restaurant",
            "school",
            "hospital",
            "lodging",
            "police",
            "shopping_mall",
        ],
        help="List of POI types to include (e.g., restaurant school pharmacy)",
    )
    args = parser.parse_args()

    run_all(poi_type_list=args.poi_type_list)
