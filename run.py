"""
Pipeline runner script.

This module executes the data cleaning function and prepares the dataset
for downstream unsupervised and supervised learning tasks.
"""

import logging
from pathlib import Path

from src.analysis.ablation_analysis import perform_ablation_analysis
from src.analysis.tsne_visualize import generate_plot_tsne_clusters
from src.analysis.unsupervised_kmeans import perform_kmeans
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
from src.perform_sensitivity_analysis import (
    plot_sensitivity_2d,
    plot_sensitivity_3d,
)
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
    df_kmeans, x_scaled = perform_kmeans(df_merged_data)
    generate_plot_tsne_clusters(df_kmeans, x_scaled)
    detect_outliers_generate_plots(df_merged_data)

    # Supervised learning analysis
    df_single_trans = df_merged_data[df_merged_data["noOfUnits"] == 1]
    df_train, df_test = split_time_series_train_test(df_single_trans)

    # --- All models and features ---
    logger.info("---Running Best Model Cross Validation")
    best_pipeline, best_result = run_time_series_cv(
        df_train,
        mode="find_best_model",
        feature_set="all_features",
        output_path=OUTPUT_PATHS["all_models_results"],
    )
    logger.info("Completed Best Model Cross Validation\n")

    # --- Best model feature ablation analysis ---
    logger.info("---Running Feature Ablation Analysis")
    # Set ablation analysis
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

    # Additive ablation analysis
    perform_ablation_analysis(
        best_pipeline, df_train, df_test, target_column="target_price"
    )
    logger.info("Completed Feature Ablation Analysis\n")

    # --- Best model sensitivity analysis ---
    logger.info("---Running Sensitivity Analysis")
    _, sensitivity_result = run_time_series_cv(
        df_train,
        mode="best_model_multi_params",
        feature_set="all_features",
        output_path=OUTPUT_PATHS["sensitivity_results"],
        all_scores=True,
    )

    plot_sensitivity_2d(
        sensitivity_result,
        metric="MAE",
        fixed_param="min_samples_leaf",
        fixed_value=5,
    )
    plot_sensitivity_2d(
        sensitivity_result,
        metric="MAE",
        fixed_param="max_depth",
        fixed_value=15,
    )
    plot_sensitivity_3d(sensitivity_result, metric="MAE", fold_idx=-1)
    logger.info("Completed Sensitivity Analysis\n")

    # --- Best model failure analysis ---
    idx_top_n_errors = [
        10137,
        9583,
        9589,
        9585,
        4576,
        4549,
        6700,
        9536,
        5411,
        4306,
    ]
    perform_failure_analysis(
        best_pipeline, df_train, df_test, indices=idx_top_n_errors
    )


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
